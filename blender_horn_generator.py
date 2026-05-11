"""
Omarindigo's Procedural Horn Generator
A Blender addon for creating customizable procedural horns.

Install: Edit > Preferences > Add-ons > Install > select this file
"""

import bpy
import bpy.props
import math
from mathutils import Vector
from bpy.types import Operator, Panel

bl_info = {
    "name": "Procedural Horn Generator",
    "author": "Omarindigo",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Horn Generator",
    "description": "Generate customizable procedural horns with control over curvature, twist, and proportions",
    "category": "Add Mesh",
}


class HORN_OT_generate(Operator):
    """Generate a procedural horn mesh"""
    bl_idname = "mesh.horn_generate"
    bl_label = "Generate Horn"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.horn_props
        
        verts, faces = create_horn_mesh(
            length=props.length,
            base_radius=props.base_radius,
            tip_radius=props.tip_radius,
            twist=props.twist,
            curve_amount=props.curve_amount,
            curve_direction=props.curve_direction,
            segments=props.segments,
            ring_segments=props.ring_segments,
            taper=props.taper,
            twist_along_length=props.twist_along_length
        )
        
        name = f"Horn_{props.length}_{props.base_radius}_{props.curve_amount}"
        create_mesh_from_data(context, verts, faces, name)
        
        return {'FINISHED'}


class HORN_OT_generate_pair(Operator):
    """Generate a pair of symmetric horns"""
    bl_idname = "mesh.horn_generate_pair"
    bl_label = "Generate Pair"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.horn_props
        
        # Left horn
        verts_l, faces_l = create_horn_mesh(
            length=props.length,
            base_radius=props.base_radius,
            tip_radius=props.tip_radius,
            twist=props.twist,
            curve_amount=props.curve_amount,
            curve_direction=-props.curve_direction,
            segments=props.segments,
            ring_segments=props.ring_segments,
            taper=props.taper,
            twist_along_length=props.twist_along_length,
            mirror_z=True
        )
        
        # Right horn
        verts_r, faces_r = create_horn_mesh(
            length=props.length,
            base_radius=props.base_radius,
            tip_radius=props.tip_radius,
            twist=props.twist,
            curve_amount=props.curve_amount,
            curve_direction=props.curve_direction,
            segments=props.segments,
            ring_segments=props.ring_segments,
            taper=props.taper,
            twist_along_length=props.twist_along_length,
            mirror_z=False
        )
        
        create_mesh_from_data(context, verts_l, faces_l, "Horn_Left")
        create_mesh_from_data(context, verts_r, faces_r, "Horn_Right")
        
        return {'FINISHED'}


class HORN_OT_add_horns_to_mask(Operator):
    """Add horns to an existing mask mesh"""
    bl_idname = "mesh.horns_to_mask"
    bl_label = "Add Horns to Mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.horn_props
        obj = context.active_object
        
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select a mesh object first")
            return {'CANCELLED'}
        
        verts_obj, faces_obj = obj.data.vertices[:], [[i for i in p.edge_keys] for p in obj.data.polygons]
        
        verts_l, faces_l = create_horn_mesh(
            length=props.length,
            base_radius=props.base_radius,
            tip_radius=props.tip_radius,
            twist=props.twist,
            curve_amount=props.curve_amount,
            curve_direction=-props.curve_direction,
            segments=props.segments,
            ring_segments=props.ring_segments,
            taper=props.taper,
            twist_along_length=props.twist_along_length,
            mirror_z=True,
            offset=Vector((props.mask_offset_x, props.mask_offset_y, props.mask_offset_z))
        )
        
        create_mesh_from_data(context, verts_l, faces_l, "Horn_Left_Attached")
        
        return {'FINISHED'}


def create_horn_mesh(
    length=2.0,
    base_radius=0.3,
    tip_radius=0.05,
    twist=0.0,
    curve_amount=0.5,
    curve_direction=1.0,
    segments=20,
    ring_segments=12,
    taper=0.3,
    twist_along_length=False,
    mirror_z=False,
    offset=Vector((0, 0, 0))
):
    """Generate horn mesh vertices and faces"""
    
    verts = []
    faces = []
    
    for i in range(segments + 1):
        t = i / segments
        
        z = t * length
        
        current_base_radius = base_radius * (1 - t * (1 - taper))
        current_tip_radius = tip_radius
        radius = max(current_tip_radius, current_base_radius * (1 - t * 0.7))
        
        curve_z = t * length
        curve_offset = math.sin(t * math.pi) * curve_amount * curve_direction
        
        if mirror_z:
            curve_offset = -curve_offset
        
        twist_amount = twist * t
        if twist_along_length:
            twist_amount = twist * (t ** 2)
        
        for j in range(ring_segments):
            angle = (2 * math.pi * j / ring_segments) + twist_amount
            
            x = math.cos(angle) * radius + curve_offset
            y = math.sin(angle) * radius
            
            verts.append(Vector((x, y, z)) + offset)
    
    for i in range(segments):
        for j in range(ring_segments):
            current = i * ring_segments + j
            next_ring = (i + 1) * ring_segments + j
            next_segment = i * ring_segments + (j + 1) % ring_segments
            next_both = (i + 1) * ring_segments + (j + 1) % ring_segments
            
            faces.append([current, next_ring, next_both, next_segment])
    
    return verts, faces


def create_mesh_from_data(context, verts, faces, name):
    """Create a mesh object from vertex and face data"""
    
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj


class HORN_PT_panel(Panel):
    """Horn Generator Panel"""
    bl_label = "Horn Generator"
    bl_idname = "HORN_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Horn Generator'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.horn_props
        
        box = layout.box()
        box.label(text="Basic Shape", icon='CURVE_DATA')
        box.prop(props, "length")
        box.prop(props, "base_radius")
        box.prop(props, "tip_radius")
        box.prop(props, "taper")
        
        box = layout.box()
        box.label(text="Curvature", icon='CURVE_BEZCURVE')
        box.prop(props, "curve_amount")
        box.prop(props, "curve_direction")
        box.prop(props, "twist")
        box.prop(props, "twist_along_length")
        
        box = layout.box()
        box.label(text="Resolution", icon='GRID')
        box.prop(props, "segments")
        box.prop(props, "ring_segments")
        
        box = layout.box()
        box.label(text="Mask Attachment", icon='MESH_MASK')
        box.prop(props, "mask_offset_x")
        box.prop(props, "mask_offset_y")
        box.prop(props, "mask_offset_z")
        
        layout.separator()
        layout.operator("mesh.horn_generate", icon='ADD')
        layout.operator("mesh.horn_generate_pair", icon='ADD')
        layout.operator("mesh.horns_to_mask", icon='MESH_DATA')


class HornProperties(bpy.types.PropertyGroup):
    """Horn generator properties"""
    length: bpy.props.FloatProperty(
        name="Length",
        default=2.0,
        min=0.1,
        max=10.0,
        description="Horn length"
    )
    base_radius: bpy.props.FloatProperty(
        name="Base Radius",
        default=0.3,
        min=0.05,
        max=1.0,
        description="Radius at the base"
    )
    tip_radius: bpy.props.FloatProperty(
        name="Tip Radius",
        default=0.02,
        min=0.001,
        max=0.5,
        description="Radius at the tip"
    )
    taper: bpy.props.FloatProperty(
        name="Taper",
        default=0.3,
        min=0.0,
        max=1.0,
        description="How quickly the horn tapers (0=constant, 1=sharp taper)"
    )
    twist: bpy.props.FloatProperty(
        name="Twist",
        default=0.0,
        min=-math.pi,
        max=math.pi,
        description="Amount of twist along the horn"
    )
    curve_amount: bpy.props.FloatProperty(
        name="Curve",
        default=0.5,
        min=-3.0,
        max=3.0,
        description="How much the horn curves"
    )
    curve_direction: bpy.props.FloatProperty(
        name="Curve Direction",
        default=1.0,
        min=-1.0,
        max=1.0,
        description="Direction of the curve (-1 = left, 1 = right)"
    )
    twist_along_length: bpy.props.BoolProperty(
        name="Twist Along Length",
        default=False,
        description="Twist intensifies toward the tip"
    )
    segments: bpy.props.IntProperty(
        name="Length Segments",
        default=20,
        min=3,
        max=100,
        description="Number of segments along the length"
    )
    ring_segments: bpy.props.IntProperty(
        name="Ring Segments",
        default=12,
        min=3,
        max=32,
        description="Number of segments around the horn"
    )
    mask_offset_x: bpy.props.FloatProperty(
        name="Offset X",
        default=0.0,
        min=-5.0,
        max=5.0,
        description="X offset for mask attachment"
    )
    mask_offset_y: bpy.props.FloatProperty(
        name="Offset Y",
        default=0.0,
        min=-5.0,
        max=5.0,
        description="Y offset for mask attachment"
    )
    mask_offset_z: bpy.props.FloatProperty(
        name="Offset Z",
        default=0.0,
        min=-5.0,
        max=5.0,
        description="Z offset for mask attachment"
    )


classes = (
    HORN_OT_generate,
    HORN_OT_generate_pair,
    HORN_OT_add_horns_to_mask,
    HORN_PT_panel,
    HornProperties,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.horn_props = bpy.props.PointerProperty(type=HornProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.horn_props


if __name__ == "__main__":
    register()
