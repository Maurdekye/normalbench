#![feature(portable_simd)]
#![feature(array_chunks)]
use std::{
    error::Error,
    simd::{
        Simd,
        num::{SimdInt, SimdUint},
    },
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::*},
    },
    thread,
    time::Instant,
    usize,
};

use clap::{Parser, crate_authors, crate_name};
use ggez::{
    ContextBuilder,
    conf::{WindowMode, WindowSetup},
    event,
    graphics::{DrawParam, Mesh, Text},
};
use ggez::{
    event::EventHandler,
    graphics::{Canvas, Color},
};
use rand::{Rng, rng};

const CHUNKSIZE: usize = 10_000;
const SIMD_CHUNKWIDTH: usize = 32;
const MEM_SIMD_CHUNKWIDTH: usize = 4;

#[derive()]
struct Renderer {
    stopping: Arc<AtomicBool>,
    mem: Arc<Mutex<Mem>>,
    args: Args,
    start_time: Instant,
}

impl Renderer {
    fn new(stopping: Arc<AtomicBool>, mem: Arc<Mutex<Mem>>, args: Args) -> Self {
        Self {
            stopping,
            mem,
            args,
            start_time: Instant::now(),
        }
    }
}

impl EventHandler for Renderer {
    fn update(&mut self, _ctx: &mut ggez::Context) -> Result<(), ggez::GameError> {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut ggez::Context) -> Result<(), ggez::GameError> {
        let height = ctx.gfx.window().inner_size().height as f32;
        let mut canvas = Canvas::from_frame(ctx, Color::WHITE);
        let mem = self.mem.lock().unwrap();

        // histogram
        for (x, value) in mem.mem.iter().enumerate() {
            if *value > 0 {
                let y = ((*value as f32) / (mem.highest as f32)) * height * 0.75;
                let line = Mesh::new_line(
                    ctx,
                    &[[x as f32, height], [x as f32, height - y]],
                    1.0,
                    Color::BLACK,
                )?;
                canvas.draw(&line, DrawParam::default());
            }
        }

        // corner stats
        let steps_per_sec =
            (self.args.steps as f64 * mem.total as f64) / self.start_time.elapsed().as_secs_f64();
        let format_option = |x: Option<usize>| x.map(|v| format!("{v}")).unwrap_or("N/A".into());
        let format_num = |n| match n {
            _ if n > 1e12 => format!("{:.3}T", n / 1e12),
            _ if n > 1e9 => format!("{:.3}B", n / 1e9),
            _ if n > 1e6 => format!("{:.3}M", n / 1e6),
            _ if n > 1e3 => format!("{:.3}K", n / 1e3),
            _ => format!("{:.3}", n),
        };
        let stats_text = format!(
            "Total: {}
Height: {}
Min: {}
Max: {}
Steps/sec: {}",
            mem.total,
            mem.highest,
            format_option(mem.min),
            format_option(mem.max),
            format_num(steps_per_sec)
        );
        let stats_text = Text::new(stats_text);
        let margin: f32 = 10.0;
        let stats_bounds = stats_text.measure(ctx)?;
        canvas.draw(
            &stats_text,
            DrawParam::default()
                .dest([margin, margin])
                .color(Color::BLACK),
        );

        // tooltip
        let pos = ctx.mouse.position();
        let x = pos.x as usize;
        if x < mem.mem.len()
            && !(pos.x < stats_bounds.x + margin && pos.y < stats_bounds.y + margin)
        {
            let value = mem.mem[x];
            let percent = if mem.highest > 0 {
                (value as f32) / (mem.highest as f32) * 100.0
            } else {
                0.0
            };
            let text = Text::new(format!("x: {}\ntotal: {}\n{:.2}%", x, value, percent));
            let draw_pos = [pos.x + 10.0, pos.y + 10.0];
            canvas.draw(
                &text,
                DrawParam::default().dest(draw_pos).color(Color::BLACK),
            );
        }

        canvas.finish(ctx)
    }

    fn quit_event(&mut self, _ctx: &mut ggez::Context) -> Result<bool, ggez::GameError> {
        self.stopping.store(true, Relaxed);
        Ok(false)
    }
}

struct Mem {
    mem: Vec<usize>,
    total: usize,
    highest: usize,
    min: Option<usize>,
    max: Option<usize>,
}

#[derive(Parser, Clone)]
struct Args {
    #[arg(short, long, default_value_t = 32768)]
    steps: usize,
    #[arg(short, long, default_value_t = 1024)]
    width: usize,
    #[arg(short, long)]
    threads: Option<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mem = Arc::new(Mutex::new(Mem {
        mem: vec![0usize; args.width],
        total: 0,
        highest: 0,
        min: None,
        max: None,
    }));
    let stopping = Arc::new(AtomicBool::new(false));

    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let threads = threads.min(args.threads.unwrap_or(threads));

    thread::scope(|scope| {
        (0..threads).for_each(|_| {
            scope.spawn(|| {
                let mut local_mem = vec![0usize; args.width];
                let mut rng = rng();
                let one = Simd::from([1; SIMD_CHUNKWIDTH]);
                let two = Simd::from([2; SIMD_CHUNKWIDTH]);
                while !stopping.load(Relaxed) {
                    for _ in 0..CHUNKSIZE {
                        let mut sim_block = Simd::from([0i16; SIMD_CHUNKWIDTH]);
                        for _ in 0..args.steps {
                            let bits: [i16; SIMD_CHUNKWIDTH] = rng.random();
                            let mut bits_simd = Simd::from(bits);
                            bits_simd = ((bits_simd.signum() * two) + one).signum();
                            sim_block += bits_simd;
                        }
                        for value in sim_block.to_array() {
                            let loc = (value + (local_mem.len()) as i16) / 2;
                            if loc >= 0 {
                                if let Some(entry) = local_mem.get_mut(loc as usize) {
                                    *entry += 1;
                                }
                            }
                        }
                    }
                    let mut mem = mem.lock().unwrap();
                    let mut highest = mem.highest;
                    for (mem_chunk, local_mem_chunk) in mem
                        .mem
                        .array_chunks_mut::<MEM_SIMD_CHUNKWIDTH>()
                        .zip(local_mem.array_chunks_mut::<MEM_SIMD_CHUNKWIDTH>())
                    {
                        let mut mem_simd: Simd<usize, MEM_SIMD_CHUNKWIDTH> =
                            Simd::from_slice(mem_chunk);
                        let local_mem_simd: Simd<usize, MEM_SIMD_CHUNKWIDTH> =
                            Simd::from_slice(local_mem_chunk);
                        mem_simd += local_mem_simd;
                        highest = mem_simd.reduce_max().max(highest);
                        mem_chunk.copy_from_slice(&mem_simd.to_array());
                        local_mem_chunk.fill(0);
                    }
                    mem.highest = highest;
                    mem.total += CHUNKSIZE * SIMD_CHUNKWIDTH;
                    let mut min = mem.min;
                    let mut max = mem.max;
                    for i in mem
                        .mem
                        .iter()
                        .enumerate()
                        .filter_map(|(i, x)| (*x > 0).then_some(i))
                    {
                        min = match min {
                            None => Some(i),
                            Some(old_min) => Some(old_min.min(i)),
                        };
                        max = match max {
                            None => Some(i),
                            Some(old_max) => Some(old_max.max(i)),
                        };
                    }
                    mem.min = min;
                    mem.max = max;
                }
            });
        });

        let (ctx, events) = ContextBuilder::new(crate_name!(), crate_authors!())
            .window_mode(
                WindowMode::default().dimensions(args.width as f32, (args.width / 2) as f32),
            )
            .window_setup(WindowSetup::default().vsync(true))
            .build()
            .unwrap();

        ctx.gfx.set_window_title("normalbench");

        let renderer = Renderer::new(Arc::clone(&stopping), Arc::clone(&mem), args.clone());

        event::run(ctx, events, renderer);
    });

    Ok(())
}
