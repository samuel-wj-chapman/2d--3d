import bpy
import os

# Clear all data
bpy.ops.wm.read_factory_settings(use_empty=True)

# Set the render resolution
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512

# Load your object
object_path = "/path/to/your/object.stl"  # Adjust this to your object path
object_name = os.path.basename(object_path).split('.')[0]
bpy.ops.import_mesh.stl(filepath=object_path)

# Assuming the object is selected after import, store its reference
obj = bpy.context.selected_objects[0]

# Set up rendering of the object

# Add a camera
bpy.ops.object.camera_add(location=(0, -10, 0))
camera = bpy.context.object

# Add a light source
bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 5))

# Set the render path
render_folder = "/path/to/save/rendered/images"  # Adjust this to your save path
if not os.path.exists(render_folder):
    os.makedirs(render_folder)

# Number of rotations per axis
num_rotations = 10  # for example, 10 rotations for each axis, adjust as needed

# Calculate rotation angles
angle_step = 360 / num_rotations

# Rotate, render, and save images
for x in range(num_rotations):
    for y in range(num_rotations):
        for z in range(num_rotations):
            # Rotate object
            obj.rotation_euler = (x * angle_step, y * angle_step, z * angle_step)
            
            # Render image
            file_name = f"{object_name}_{x * angle_step}_{y * angle_step}_{z * angle_step}.png"
            bpy.context.scene.render.filepath = os.path.join(render_folder, file_name)
            bpy.ops.render.render(write_still=True)

print("Rendering completed!")
