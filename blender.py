import bpy
import os
import math
import random

# Parameters
stl_directory = "path/to/stl_files"  # Update with actual path
stl_file_paths = [os.path.join(stl_directory, f) for f in os.listdir(stl_directory) if f.endswith('.stl')]
output_directory = "path/to/output"  # Update with actual path
num_images_per_object = 10
view_coverage_range = (0.7, 0.9)  # Object coverage in the camera view

# Function to delete all objects in the scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

# Function to load an STL file
def load_stl(file_path):
    bpy.ops.import_mesh.stl(filepath=file_path)

# Function to position the camera based on object bounds
def position_camera(obj, coverage):
    # Calculate the camera distance from the object
    dimensions = obj.dimensions
    max_dimension = max(dimensions)
    camera_distance = (0.5 * max_dimension) / math.tan(math.radians(25))  # 25 is half the camera's FOV
    coverage_factor = random.uniform(*coverage)
    camera_distance /= coverage_factor

    # Position the camera
    bpy.context.scene.camera.location = (0, -camera_distance, 0)

# Function to render and save an image
def render_save_image(obj_name, angle_x, angle_y, output_path, image_id):
    # Set object rotation
    bpy.context.scene.objects[obj_name].rotation_euler = (math.radians(angle_x), 0, math.radians(angle_y))

    # Update scene
    bpy.context.view_layer.update()

    # Render and save image
    file_name = f"{obj_name}_angleX{angle_x}_angleY{angle_y}_{image_id}.png"
    bpy.context.scene.render.filepath = os.path.join(output_path, file_name)
    bpy.ops.render.render(write_still=True)

# Setup lighting
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 10))

# Main processing loop
for stl_path in stl_file_paths:
    clear_scene()
    load_stl(stl_path)
    obj = bpy.context.selected_objects[0]  # Assuming STL has one object
    position_camera(obj, view_coverage_range)

    for i in range(num_images_per_object):
        angle_x = random.randint(0, 360)
        angle_y = random.randint(0, 360)
        render_save_image(obj.name, angle_x, angle_y, output_directory, i)

print("Dataset generation complete.")
