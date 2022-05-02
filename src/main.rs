use std::collections::HashMap;

use fontdue::{
    layout::{CoordinateSystem, GlyphRasterConfig, Layout, LayoutSettings, TextStyle},
    Font, FontSettings, Metrics,
};
use glam::vec2;
use miniquad::{
    conf::Conf, Bindings, BlendState, Buffer, BufferLayout, BufferType, Context, EventHandler,
    FilterMode, Pipeline, PipelineParams, Shader, Texture, TextureFormat, TextureParams,
    TextureWrap, UserData, VertexAttribute, VertexFormat,
};

use crate::shader::Vertex;

mod shader {
    use miniquad::{ShaderMeta, UniformDesc, UniformType};

    #[repr(C)]
    #[derive(Debug)]
    pub struct Vertex {
        pub pos: glam::Vec2,
        pub uv0: glam::Vec2,
        pub color: glam::Vec4,
    }

    pub const VERTEX: &str = "
    #version 140

    // Attributes
    in vec2 pos;
    in vec2 uv0;
    in vec4 color;

    // Uniforms
    uniform mat4 mvp;

    // Outputs
    out vec2 uv;
    out vec4 fragment_color;

    void main() {
        gl_Position = mvp * vec4(pos, 1.0, 1.0);
        uv = uv0;
        fragment_color = color;
    }
    ";

    pub const FRAGMENT: &str = "
    #version 140

    // Attributes
    in vec2 uv;
    in vec4 fragment_color;

    // Uniforms
    uniform sampler2D texture_atlas;

    // Outputs
    out vec4 output_color;


    #define ALPHA_THRESH 0.1

    void main() {
        vec4 sample = texture(texture_atlas, uv);
        output_color = sample * fragment_color;
        output_color.a = output_color.r > ALPHA_THRESH ? output_color.a : 0.0;
    }
    ";

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            uniforms: miniquad::UniformBlockLayout {
                uniforms: vec![UniformDesc::new("mvp", UniformType::Mat4)],
            },
            images: vec!["texture_atlas".to_string()],
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
    }
}

struct Editor {
    character_bindings: Vec<(Bindings, i32)>,
    shader: Shader,
    pipeline: Pipeline,

    fonts: [Font; 1],
    font_settings: FontSettings,
    layout: Layout<glam::Vec4>,
    glyph_cache: HashMap<GlyphRasterConfig, (Metrics, Texture)>,

    needs_redraw: bool,
    text_changed: bool,
    n: f32,

    mouse: glam::Vec2,
    scroll: glam::Vec2,
    prev_scroll: glam::Vec2,
}

const TEXT: &str = include_str!("../assets/example.txt");
const TEXT_PX_SIZE: f32 = 22.0;
const FONT_BYTES: &[u8] =
    include_bytes!("../assets/Source_Code_Pro/static/SourceCodePro-Regular.ttf");
// const FONT_BYTES: &[u8] = include_bytes!("../assets/Roboto_Mono/RobotoMono-VariableFont_wght.ttf");

impl Editor {
    pub fn from_context(ctx: &mut Context) -> Self {
        let font_settings = FontSettings {
            scale: 20.0,
            ..FontSettings::default()
        };

        let font = Font::from_bytes(FONT_BYTES, font_settings).unwrap();
        let fonts = [font];
        let layout = Layout::new(CoordinateSystem::PositiveYUp);

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();
        let pipeline = Pipeline::new(ctx, &[], &[], shader);
        ctx.set_window_size(100, 100);

        Self {
            shader,
            pipeline,

            character_bindings: vec![],

            fonts,
            font_settings,
            layout,
            glyph_cache: HashMap::new(),

            needs_redraw: true,
            text_changed: true,

            n: 0.0,

            mouse: glam::Vec2::ZERO,
            scroll: glam::Vec2::ZERO,
            prev_scroll: glam::Vec2::ZERO,
        }
    }

    pub fn update_text(&mut self, ctx: &mut miniquad::Context) {
        println!("[{:.4}] Rebuilding text layout...", self.n);

        self.layout.reset(&LayoutSettings {
            ..LayoutSettings::default()
        });

        for (n, line) in TEXT.lines().enumerate() {
            self.layout.append(
                &self.fonts,
                &TextStyle::with_user_data(
                    &format!("{:>5}   ", n + 1)[..],
                    TEXT_PX_SIZE,
                    0,
                    glam::vec4(1.0, 1.0, 1.0, 0.4),
                ),
            );

            self.layout.append(
                &self.fonts,
                &TextStyle::with_user_data(line, TEXT_PX_SIZE, 0, glam::vec4(1.0, 1.0, 1.0, 1.0)),
            );

            self.layout.append(
                &self.fonts,
                &TextStyle::with_user_data("\n", TEXT_PX_SIZE, 0, glam::vec4(1.0, 1.0, 1.0, 1.0)),
            );
        }

        let mut characters: HashMap<GlyphRasterConfig, (Texture, Vec<Vertex>, Vec<u16>)> =
            HashMap::new();

        for glyph in self.layout.glyphs() {
            let key = glyph.key;

            let (_metrics, texture) = self.glyph_cache.entry(key).or_insert_with(|| {
                let (metrics, bitmap) = self.fonts[glyph.font_index]
                    .rasterize_indexed_subpixel(glyph.key.glyph_index, TEXT_PX_SIZE);

                let texture = Texture::new(
                    ctx,
                    miniquad::TextureAccess::Static,
                    Some(&bitmap),
                    TextureParams {
                        format: TextureFormat::RGB8,
                        wrap: TextureWrap::Clamp,
                        filter: FilterMode::Linear,
                        width: metrics.width as u32,
                        height: metrics.height as u32,
                    },
                );

                (metrics, texture)
            });

            let (_the_texture, character_verticies, character_indicies) =
                characters.entry(key).or_insert((*texture, vec![], vec![]));

            let (x, y) = (glyph.x, glyph.y);
            let (w, h) = (glyph.width as f32, glyph.height as f32);

            let i = character_verticies.len() as u16;

            #[rustfmt::skip]
            character_verticies.append(&mut vec![
                Vertex { pos: glam::vec2((x + 0.0) * 0.82, (y + 0.0) * 0.82), uv0: vec2(0.0, 1.0), color: glyph.user_data },
                Vertex { pos: glam::vec2((x + 0.0) * 0.82, (y + h  ) * 0.82), uv0: vec2(0.0, 0.0), color: glyph.user_data },
                Vertex { pos: glam::vec2((x + w  ) * 0.82, (y + h  ) * 0.82), uv0: vec2(1.0, 0.0), color: glyph.user_data },
                Vertex { pos: glam::vec2((x + w  ) * 0.82, (y + 0.0) * 0.82), uv0: vec2(1.0, 1.0), color: glyph.user_data },
            ]);

            #[allow(clippy::identity_op)]
            character_indicies.append(&mut vec![i + 0, i + 1, i + 2, i + 0, i + 3, i + 2]);
        }

        self.pipeline = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv0", VertexFormat::Float2),
                VertexAttribute::new("color", VertexFormat::Float4),
            ],
            self.shader,
            PipelineParams {
                alpha_blend: Some(BlendState::new(
                    miniquad::Equation::Add,
                    miniquad::BlendFactor::Value(miniquad::BlendValue::SourceAlpha),
                    miniquad::BlendFactor::OneMinusValue(miniquad::BlendValue::SourceAlpha),
                )),
                color_blend: Some(BlendState::new(
                    miniquad::Equation::Add,
                    miniquad::BlendFactor::Value(miniquad::BlendValue::SourceAlpha),
                    miniquad::BlendFactor::One,
                )),
                ..Default::default()
            },
        );

        self.character_bindings = characters
            .into_iter()
            .map(|(_k, (texture, verticies, indicies))| {
                (
                    Bindings {
                        vertex_buffers: vec![Buffer::immutable(
                            ctx,
                            BufferType::VertexBuffer,
                            &verticies[..],
                        )],
                        index_buffer: Buffer::immutable(
                            ctx,
                            BufferType::IndexBuffer,
                            &indicies[..],
                        ),
                        images: vec![texture],
                    },
                    indicies.len() as i32,
                )
            })
            .collect::<Vec<(Bindings, i32)>>();
    }

    pub fn render(&mut self, ctx: &mut miniquad::Context) {
        self.n += 0.03;
        let t = (self.n.sin() + 1.0) / 2.0;
        let t2 = (self.n.cos() + 1.0) / 2.0;

        let (w, h) = ctx.screen_size();
        let projection =
            glam::Mat4::orthographic_rh_gl(-w / 2.0, w / 2.0, -h / 2.0, h / 2.0, 0.01, 560.0);

        let model = glam::Mat4::from_scale_rotation_translation(
            glam::vec3(1.0, 1.0, 1.0),
            glam::Quat::IDENTITY,
            glam::vec3(0.0, 0.0, 0.0),
        );
        let view = glam::Mat4::look_at_rh(
            glam::vec3(
                self.prev_scroll.x + w / 2.0,
                -self.prev_scroll.y - h / 2.0,
                10.0,
            ),
            glam::vec3(
                self.prev_scroll.x + w / 2.0,
                -self.prev_scroll.y - h / 2.0,
                0.0,
            ),
            glam::vec3(0.0, 1.0, 0.0),
        );
        let mvp = projection * view * model;

        if self.text_changed {
            self.update_text(ctx);
            self.text_changed = false;
        }

        let (r, g, b, a) = (15.0 / 255.0, 15.0 / 255.0, 15.0 / 255.0, 1.0);
        ctx.begin_default_pass(miniquad::PassAction::clear_color(r, g, b, a));

        /*
            ctx.begin_default_pass(miniquad::PassAction::clear_color(
                t * t2 * 0.3,
                (1.0 - t) * 0.3,
                t * t * t * 0.3,
                1.0,
            ));
        */

        ctx.apply_pipeline(&self.pipeline);

        ctx.apply_uniforms(&shader::Uniforms { mvp });

        for (binding, num_elements) in &self.character_bindings {
            ctx.apply_bindings(binding);
            ctx.draw(0, *num_elements, 1);
        }

        ctx.end_render_pass();
    }
}

impl EventHandler for Editor {
    fn update(&mut self, _ctx: &mut miniquad::Context) {
        let dt = 0.2;
        self.prev_scroll = (1.0 - dt) * self.prev_scroll + self.scroll * dt;

        if self.prev_scroll != self.scroll {
            self.needs_redraw = true;
        }
    }

    fn draw(&mut self, ctx: &mut miniquad::Context) {
        if self.needs_redraw {
            self.render(ctx);
            self.needs_redraw = false;
        }
    }

    fn resize_event(&mut self, _ctx: &mut Context, _width: f32, _height: f32) {
        self.needs_redraw = true;
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32) {
        if self.mouse.x != x || self.mouse.y != y {
            self.needs_redraw = true;
        }

        self.mouse = glam::vec2(x, y);
    }

    fn mouse_wheel_event(&mut self, _ctx: &mut Context, dx: f32, dy: f32) {
        let old_scroll = self.scroll;
        self.scroll -= glam::vec2(dx, dy) * 15.0;

        self.scroll.x = f32::max(0.0, self.scroll.x);
        self.scroll.y = f32::min(f32::max(0.0, self.scroll.y), self.layout.height());

        if self.scroll != old_scroll {
            self.needs_redraw = true;
        }
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        _keycode: miniquad::KeyCode,
        _keymods: miniquad::KeyMods,
        _repeat: bool,
    ) {
        self.needs_redraw = true;
        self.text_changed = true;
    }
}

fn main() {
    miniquad::start(
        Conf {
            window_title: "Erik's Editor".to_owned(),
            high_dpi: true,
            sample_count: 8,
            ..Default::default()
        },
        |mut ctx| UserData::owning(Editor::from_context(&mut ctx), ctx),
    );
}
