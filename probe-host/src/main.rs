//! Throwaway host-consumption probe for the Row-1 borrow accessor.
//!
//! NOT shippable code. It answers whether the borrowed GL name denotes a
//! texture the host can actually consume: a hidden 1x1 window provides the
//! WGL display (a windowless display is crippled by design and cannot create
//! contexts on this driver), the host writes a pattern with raw GL, imports
//! the name into wgpu-hal, borrows it back, and reads the pixels with raw
//! GL. Same thread throughout; no submit, no fences.
//!
//! Needs a real GL driver. Run from this directory: `cargo run`.
//! Paste the full output back.

use std::ffi::CString;
use std::num::NonZeroU32;

use glow::HasContext;
use glutin::config::GlConfig as _;
use glutin::context::{ContextApi, GlProfile, NotCurrentGlContext as _, Version};
use glutin::display::{GetGlDisplay as _, GlDisplay as _};
use glutin_winit::{DisplayBuilder, GlWindow as _};
use raw_window_handle::HasWindowHandle as _;
use wgpu_hal::{Adapter as _, Device as _};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoop};

struct Probe;

impl ApplicationHandler for Probe {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = winit::window::Window::default_attributes()
            .with_visible(false)
            .with_inner_size(winit::dpi::PhysicalSize::new(1, 1));

        let template = glutin::config::ConfigTemplateBuilder::new();
        let (window, gl_config) = DisplayBuilder::new()
            .with_window_attributes(Some(attrs))
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|a, c| {
                        if c.num_samples() > a.num_samples() {
                            c
                        } else {
                            a
                        }
                    })
                    .expect("FAIL: no GL config")
            })
            .expect("FAIL: display setup failed");
        let window = window.expect("FAIL: WGL always returns a window");
        let raw_win = window.window_handle().ok().map(|h| h.as_raw());
        let gl_display = gl_config.display();

        // Two attempts: Core 3.3 first, then Compatibility 3.3. Some legacy
        // drivers refuse a Core profile request while accepting Compat.
        let make_ctx = |profile: GlProfile| unsafe {
            let attrs = glutin::context::ContextAttributesBuilder::new()
                .with_profile(profile)
                .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
                .build(raw_win);
            gl_display.create_context(&gl_config, &attrs)
        };
        let not_current = make_ctx(GlProfile::Core)
            .or_else(|e1| make_ctx(GlProfile::Compatibility).map_err(|e2| (e1, e2)))
            .unwrap_or_else(|(e1, e2)| {
                println!("FAIL: GL context creation failed (core: {e1:?}, compat: {e2:?})");
                std::process::exit(7);
            });
        let surf_attrs = window
            .build_surface_attributes(Default::default())
            .expect("FAIL: surface attributes failed");
        let surface = unsafe { gl_display.create_window_surface(&gl_config, &surf_attrs) }
            .expect("FAIL: window surface creation failed");
        let _current = not_current
            .make_current(&surface)
            .expect("FAIL: cannot make GL context current");

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                gl_display.get_proc_address(&CString::new(s).unwrap()) as *const _
            })
        };
        let version = unsafe { gl.get_parameter_string(glow::VERSION) };
        println!(
            "host GL: {version} / {} / {}",
            unsafe { gl.get_parameter_string(glow::VENDOR) },
            unsafe { gl.get_parameter_string(glow::RENDERER) },
        );
        let major: u32 = version
            .split('.')
            .next()
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        assert!(
            major >= 3,
            "FAIL: GL major version {major} below the 3.x floor"
        );
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        }

        const W: i32 = 64;
        const H: i32 = 64;
        let mut pattern = vec![0u8; (W * H * 4) as usize];
        for (i, b) in pattern.iter_mut().enumerate() {
            *b = (i.wrapping_mul(5).wrapping_add(11)) as u8;
        }
        let host_name = unsafe {
            let name = gl.create_texture().expect("FAIL: host gen_texture failed");
            gl.bind_texture(glow::TEXTURE_2D, Some(name));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, W, H);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                W,
                H,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&pattern)),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            let id: NonZeroU32 = name.0;
            println!("host texture name: {id}");
            id
        };

        let exposed = unsafe {
            <wgpu_hal::api::Gles as wgpu_hal::Api>::Adapter::new_external(
                |name| gl_display.get_proc_address(&CString::new(name).unwrap()) as *const _,
                wgpu_types::GlBackendOptions::default(),
            )
        }
        .expect("FAIL: hal adapter rejected the host context");
        let opened = unsafe {
            exposed.adapter.open(
                wgpu_types::Features::empty(),
                &wgpu_types::Limits::downlevel_defaults(),
                &wgpu_types::MemoryHints::default(),
            )
        }
        .expect("FAIL: hal device open failed");

        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dropped_flag = dropped.clone();
        let hal_texture = unsafe {
            opened.device.texture_from_raw(
                host_name,
                &wgpu_hal::TextureDescriptor {
                    label: None,
                    size: wgpu_types::Extent3d {
                        width: W as u32,
                        height: H as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu_types::TextureDimension::D2,
                    format: wgpu_types::TextureFormat::Rgba8Unorm,
                    usage: wgpu_types::TextureUses::RESOURCE,
                    memory_flags: wgpu_hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                Some(Box::new(move || {
                    dropped_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                })),
            )
        };

        let borrowed = opened.device.native_texture_handle(&hal_texture);
        println!("hal borrow: {borrowed:?}");
        assert_eq!(
            borrowed.map(|t| t.0),
            Some(host_name),
            "FAIL: borrow did not return the host texture name"
        );

        let mut readback = vec![0u8; pattern.len()];
        unsafe {
            gl.finish();
            gl.bind_texture(glow::TEXTURE_2D, Some(glow::NativeTexture(host_name)));
            gl.get_tex_image(
                glow::TEXTURE_2D,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut readback)),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
        assert_eq!(readback.len(), pattern.len(), "FAIL: readback size differs");
        match readback
            .iter()
            .zip(pattern.iter())
            .position(|(a, b)| a != b)
        {
            None => println!("host readback: {} bytes identical", readback.len()),
            Some(i) => {
                println!(
                    "FAIL: pixel mismatch at byte {i}: got {}, want {}",
                    readback[i], pattern[i]
                );
                std::process::exit(4);
            }
        }

        unsafe { opened.device.destroy_texture(hal_texture) };
        assert!(
            dropped.load(std::sync::atomic::Ordering::SeqCst),
            "FAIL: drop callback did not fire on hal texture destroy"
        );
        println!("drop callback: fired, host texture not deleted by wgpu");
        unsafe {
            gl.delete_texture(glow::NativeTexture(host_name));
        }

        println!("PASS: host wrote, hal borrowed, host read back identical");
        event_loop.exit();
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("FAIL: event loop creation failed");
    let mut app = Probe;
    event_loop
        .run_app(&mut app)
        .expect("FAIL: event loop failed");
}
