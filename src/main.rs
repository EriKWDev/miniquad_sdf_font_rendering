use std::collections::HashMap;

use fontdue::{
    layout::{CoordinateSystem, GlyphRasterConfig, Layout, LayoutSettings, TextStyle},
    Font, FontSettings, Metrics,
};
use glam::vec2;
use miniquad::{
    conf::Conf, Bindings, Buffer, BufferLayout, BufferType, Context, EventHandler, FilterMode,
    Pipeline, Shader, TextureFormat, TextureWrap, UserData, VertexAttribute, VertexFormat,
};

use crate::shader::Vertex;

mod shader {
    use miniquad::{ShaderMeta, UniformDesc, UniformType};

    #[repr(C)]
    #[derive(Debug)]
    pub struct Vertex {
        pub pos: glam::Vec2,
        pub uv0: glam::Vec2,
    }

    pub const VERTEX: &str = "
    #version 140

    // Attributes
    in vec2 pos;
    in vec2 uv0;

    // Uniforms
    uniform mat4 mvp;

    // Outputs
    out vec2 uv;

    void main() {
        gl_Position = mvp * vec4(pos, 1.0, 1.0);
        uv = uv0;
    }
    ";

    pub const FRAGMENT: &str = "
    #version 140

    // Attributes
    in vec2 uv;

    // Uniforms
    uniform sampler2D texture_atlas;

    // Outputs
    out vec4 frag_color;

    void main() {
        vec4 sample = texture(texture_atlas, uv);
        frag_color = sample;
        frag_color = vec4(1.0) * vec4(uv, 1.0, 1.0);
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
    bindings: Bindings,
    pipeline: Pipeline,

    fonts: [Font; 1],
    font_settings: FontSettings,
    layout: Layout,
    glyph_cache: HashMap<GlyphRasterConfig, (Metrics, Vec<u8>)>,
    num_elements: i32,

    needs_redraw: bool,
    text_changed: bool,
    n: f32,
    mouse_x: f32,
    mouse_y: f32,

    scroll_x: f32,
    scroll_y: f32,
}

const TEXT: &str = include_str!("../assets/example.txt");

impl Editor {
    pub fn from_context(ctx: &mut Context) -> Self {
        let font = include_bytes!("../assets/Source_Code_Pro/SourceCodePro-VariableFont_wght.ttf")
            as &[u8];

        let font_settings = FontSettings {
            scale: 20.0,
            ..FontSettings::default()
        };

        let font = Font::from_bytes(font, font_settings).unwrap();
        let fonts = [font];
        let layout = Layout::new(CoordinateSystem::PositiveYUp);

        let bindings = Bindings {
            vertex_buffers: vec![],
            index_buffer: Buffer::immutable(ctx, BufferType::IndexBuffer, &[0]),
            images: vec![],
        };

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();
        let pipeline = Pipeline::new(ctx, &[], &[], shader);

        Self {
            bindings,
            pipeline,

            fonts,
            font_settings,
            layout,
            glyph_cache: HashMap::new(),

            needs_redraw: true,
            text_changed: true,

            n: 0.0,

            mouse_x: 0.0,
            mouse_y: 0.0,

            scroll_x: 0.0,
            scroll_y: 0.0,

            num_elements: 0,
        }
    }

    pub fn update_text(&mut self, ctx: &mut miniquad::Context) {
        println!("[{:.4}] Rebuilding text layout...", self.n);

        self.layout.reset(&LayoutSettings {
            ..LayoutSettings::default()
        });

        self.layout
            .append(&self.fonts, &TextStyle::new(TEXT, 35.0, 0));

        let mut images_in_total_image: HashMap<GlyphRasterConfig, (Metrics, Vec<u8>)> =
            HashMap::new();

        let num_glyps = self.layout.glyphs().len();

        let mut verticies = Vec::<shader::Vertex>::with_capacity(num_glyps * 4); // 4 verticies per quad
        let mut indicies = Vec::<u32>::with_capacity(num_glyps * 6); // 3 indicies per triangle

        let (mut largest_glyph_width, mut largest_glyph_height) = (0, 0);

        for glyph in self.layout.glyphs() {
            let key = glyph.key;

            let (metrics, bitmap) = self.glyph_cache.entry(key).or_insert_with(|| {
                self.fonts[glyph.font_index].rasterize_indexed_subpixel(glyph.key.glyph_index, 35.0)
            });

            images_in_total_image
                .entry(key)
                .or_insert((*metrics, bitmap.to_vec()));

            largest_glyph_width = usize::max(largest_glyph_width, metrics.width);
            largest_glyph_height = usize::max(largest_glyph_height, metrics.height);

            let (x, y) = (glyph.x, glyph.y);
            let (w, h) = (glyph.width as f32, glyph.height as f32);

            #[rustfmt::skip]
            verticies.append(&mut vec![
                Vertex { pos: glam::vec2(x + 0.0, y + 0.0), uv0: vec2(0.0, 1.0) },
                Vertex { pos: glam::vec2(x + 0.0, y + h  ), uv0: vec2(0.0, 0.0) },
                Vertex { pos: glam::vec2(x + w,   y + h  ), uv0: vec2(1.0, 0.0) },
                Vertex { pos: glam::vec2(x + w,   y + 0.0), uv0: vec2(1.0, 1.0) }
            ]);

            let i = verticies.len() as u32;

            #[rustfmt::skip]
            #[allow(clippy::identity_op)]
            indicies.append(&mut vec![
                i + 0, i + 1, i + 2,
                i + 0, i + 3, i + 2,
            ]);
        }

        let image_bytes =
            vec![
                1_u8;
                images_in_total_image.len() * largest_glyph_width * largest_glyph_height * 3
            ];

        self.num_elements = indicies.len() as i32;

        let shader = Shader::new(ctx, shader::VERTEX, shader::FRAGMENT, shader::meta()).unwrap();
        self.pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float2),
                VertexAttribute::new("uv0", VertexFormat::Float2),
            ],
            shader,
        );

        self.bindings = Bindings {
            vertex_buffers: vec![Buffer::immutable(
                ctx,
                BufferType::VertexBuffer,
                &verticies[..],
            )],
            index_buffer: Buffer::immutable(ctx, BufferType::IndexBuffer, &indicies[..]),
            images: vec![miniquad::Texture::new(
                ctx,
                miniquad::TextureAccess::Static,
                Some(&image_bytes[..]),
                miniquad::TextureParams {
                    format: TextureFormat::RGB8,
                    wrap: TextureWrap::Repeat, // TODO: Change to clamp after debug
                    filter: FilterMode::Linear,
                    width: largest_glyph_width as u32,
                    height: largest_glyph_height as u32 * images_in_total_image.len() as u32,
                },
            )],
        };
    }

    pub fn render(&mut self, ctx: &mut miniquad::Context) {
        self.n += 0.03;
        let t = (self.n.sin() + 1.0) / 2.0;
        let t2 = (self.n.cos() + 1.0) / 2.0;

        let (w, h) = ctx.screen_size();
        let projection =
            glam::Mat4::orthographic_rh_gl(-w / 2.0, w / 2.0, -h / 2.0, h / 2.0, 0.01, 560.0);

        let model = glam::Mat4::from_scale_rotation_translation(
            glam::vec3(0.6, 0.6, 0.6),
            glam::Quat::IDENTITY,
            glam::vec3(0.0, 0.0, 0.0),
        );
        let view = glam::Mat4::look_at_rh(
            glam::vec3(self.scroll_x + w / 2.0, -self.scroll_y - h / 2.0, 10.0),
            glam::vec3(self.scroll_x + w / 2.0, -self.scroll_y - h / 2.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );
        let mvp = projection * view * model;

        let view2 = glam::Mat4::look_at_rh(
            glam::vec3(0.0, 0.0, 10.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );
        let model2 = glam::Mat4::from_scale_rotation_translation(
            glam::vec3(0.1, 0.1, 0.1),
            glam::Quat::IDENTITY,
            glam::vec3(w / 2.0 - 200.0, h / 2.0, 0.0),
        );
        let mvp2 = projection * view2 * model2;

        if self.text_changed {
            self.update_text(ctx);
            self.text_changed = false;
        }

        ctx.begin_default_pass(miniquad::PassAction::clear_color(
            t * t2,
            1.0 - t,
            t * t * t,
            1.0,
        ));

        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.apply_uniforms(&shader::Uniforms { mvp });
        ctx.draw(0, self.num_elements, 1);

        ctx.apply_uniforms(&shader::Uniforms { mvp: mvp2 });
        ctx.draw(0, self.num_elements, 1);
        ctx.end_render_pass();
    }
}

impl EventHandler for Editor {
    fn update(&mut self, _ctx: &mut miniquad::Context) {}

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
        if self.mouse_x != x || self.mouse_y != y {
            self.needs_redraw = true;
        }

        self.mouse_x = x;
        self.mouse_y = y;
    }

    fn mouse_wheel_event(&mut self, _ctx: &mut Context, dx: f32, dy: f32) {
        self.scroll_x -= dx * 4.0;
        self.scroll_y -= dy * 4.0;

        self.scroll_x = f32::max(0.0, self.scroll_x);
        self.scroll_y = f32::max(0.0, self.scroll_y);

        self.needs_redraw = true;
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
    miniquad::start(Conf::default(), |mut ctx| {
        UserData::owning(Editor::from_context(&mut ctx), ctx)
    });
}
