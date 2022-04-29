use std::collections::HashMap;

use fontdue::{
    layout::{CoordinateSystem, GlyphRasterConfig, Layout, LayoutSettings, TextStyle},
    Font, FontSettings, Metrics,
};
use glam::{vec2, vec3};
use miniquad::{
    conf::Conf, Bindings, Buffer, BufferType, Context, EventHandler, Pipeline, Shader, UserData,
};

mod shader {
    use miniquad::ShaderMeta;

    #[repr(C)]
    pub struct InstanceData {
        pub uv: glam::Vec2,
        pub mvp: glam::Mat4,
    }

    pub const VERTEX: &str = "
    #version 140

    in vec2 pos;

    uniform struct InstanceData {
        mat4 mvp;
    } instance_data[100];

    void main() {
        gl_Position = instance_data[gl_InstanceID].mvp * vec4(pos, 0.0, 1.0);
    }
    ";

    pub const FRAGMENT: &str = "
    #version 140

    out vec4 frag_color;

    void main() {
        frag_color = vec4(1.0);
    }
    ";

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            uniforms: miniquad::UniformBlockLayout { uniforms: vec![] },
            images: vec![],
        }
    }
}

struct Editor {
    bindings: Bindings,
    pipeline: Pipeline,

    fonts: [Font; 1],
    font_settings: FontSettings,
    layout: Layout,
    glyph_cache: HashMap<GlyphRasterConfig, (Metrics, Vec<Vec<[u8; 3]>>)>,

    needs_redraw: bool,
    n: f32,
    mouse_x: f32,
    mouse_y: f32,
}

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
        let layout = Layout::new(CoordinateSystem::PositiveYDown);

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
            n: 0.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    pub fn update_text(&mut self, ctx: &mut miniquad::Context) {
        dbg!("Rebuilding text layout...");

        let text = include_str!("../assets/example.txt");

        self.layout.reset(&LayoutSettings {
            ..LayoutSettings::default()
        });

        self.layout
            .append(&self.fonts, &TextStyle::new(text, 35.0, 0));

        let mut total_image = Vec::<Vec<[u8; 3]>>::new();

        let verticies: [glam::Vec2; 4] = [
            vec2(0.0, 0.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0),
        ];
        let indicies: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let mut instance_data = Vec::<shader::InstanceData>::new();

        for glyph in self.layout.glyphs() {
            let key = glyph.key;

            let (metrics, image) = self.glyph_cache.entry(key).or_insert_with(|| {
                let (metrics, bitmap) = self.fonts[glyph.font_index]
                    .rasterize_indexed_subpixel(glyph.key.glyph_index, 35.0);

                let mut image = vec![vec![[0_u8; 3]; metrics.height]; metrics.width];

                for (i, _value) in bitmap.iter().step_by(3).enumerate() {
                    let x = (i) % metrics.width;
                    let y = (i) / metrics.width;

                    image[x][y] = [bitmap[i], bitmap[i + 1], bitmap[i + 2]];
                }

                (metrics, image)
            });

            instance_data.push(shader::InstanceData {
                uv: vec2(0.0, 0.0),
                mvp: glam::Mat4::from_scale_rotation_translation(
                    vec3(1.0, 1.0, 0.0),
                    glam::Quat::IDENTITY,
                    vec3(0.0, 0.0, 0.0),
                ),
            });
        }

        self.bindings = Bindings {
            vertex_buffers: vec![Buffer::immutable(ctx, BufferType::VertexBuffer, &verticies)],
            index_buffer: Buffer::immutable(ctx, BufferType::IndexBuffer, &indicies),
            images: vec![],
        };
    }

    pub fn render(&mut self, ctx: &mut miniquad::Context) {
        self.n += 0.03;
        let t = (self.n.sin() + 1.0) / 2.0;

        let projection = glam::Mat4::orthographic_rh_gl(0.0, 0.0, 0.0, 0.0, 0.01, 10.0);

        self.update_text(ctx);

        ctx.begin_default_pass(miniquad::PassAction::clear_color(t, 0.0, 1.0 - t, 1.0));
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.draw(0, 6, 100);
        ctx.end_render_pass();
    }
}

impl EventHandler for Editor {
    fn update(&mut self, _ctx: &mut miniquad::Context) {}

    fn draw(&mut self, ctx: &mut miniquad::Context) {
        if self.needs_redraw {
            self.needs_redraw = false;
            self.render(ctx);
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

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        _keycode: miniquad::KeyCode,
        _keymods: miniquad::KeyMods,
        _repeat: bool,
    ) {
        self.needs_redraw = true;
    }
}

fn main() {
    miniquad::start(Conf::default(), |mut ctx| {
        UserData::owning(Editor::from_context(&mut ctx), ctx)
    });
}
