iimport bpy
import os
import math
import random

import os

# Use os.path.expanduser to expand the tilde to the full home directory path
stl_directory = os.path.expanduser("~/Documents/dataset/stldataset")
stl_file_paths = [os.path.join(stl_directory, f) for f in os.listdir(stl_directory) if f.endswith('.stl')]
output_directory = os.path.expanduser("~/Documents/dataset/images2")
num_images_per_object = 500
view_coverage_range = (0.7, 0.9)  # Object coverage in the camera view


# Set the render resolution to 224x224 pixels
bpy.context.scene.render.resolution_x = 600
bpy.context.scene.render.resolution_y = 600
bpy.context.scene.render.resolution_percentage = 100

def set_origin_to_center_of_geometry(obj):
    # Set the object as the active object
    bpy.context.view_layer.objects.active = obj
    # Select the object
    obj.select_set(True)
    # Set the origin to the center of geometry
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    # Deselect the object
    obj.select_set(False)

# Function to delete all objects in the scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Function to ensure a camera exists and is set as the active camera for the scene
def setup_camera():
    # Check if there is a camera in the scene
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    
    # If no camera was found, create one
    if camera is None:
        bpy.ops.object.camera_add(location=(0, -3, 1))
        camera = bpy.context.object
    
    # Set the newly added or found camera as the active camera
    bpy.context.scene.camera = camera

    return camera

def resize_object(obj, target_max_dimension):
    # Calculate the current maximum dimension of the object
    current_max_dimension = max(obj.dimensions)

    # Calculate the scale factor
    scale_factor = target_max_dimension / current_max_dimension

    # Apply the scale factor uniformly to the object
    obj.scale *= scale_factor


def setup_lights():
    light_data = bpy.data.lights.new(name="Soft_Light", type='AREA')
    light_data.energy = 1000  # Adjust energy as needed for your scene

    for angle in range(0, 360, 45):  # Creates lights in a circular pattern
        light_object = bpy.data.objects.new(name="Soft_Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (10 * math.sin(math.radians(angle)), 10 * math.cos(math.radians(angle)), 10)
        light_object.rotation_euler = (math.radians(45), 0, math.radians(angle))


# Function to load an STL file
def load_stl(file_path):
    bpy.ops.import_mesh.stl(filepath=file_path)
    
def look_at(obj):
    # Pointing the camera to the object
    direction = obj.location - bpy.context.scene.camera.location
    # Rotation angles
    rot_quat = direction.to_track_quat('-Z', 'Y')
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

def position_camera(obj, fixed_distance):
    # Random angles for spherical coordinates
    theta = math.radians(random.uniform(0, 360))  # Random angle in [0, 360)
    phi = math.radians(random.uniform(0, 180))   # Random angle in [0, 180)

    # Spherical to Cartesian conversion for camera position
    x = fixed_distance * math.sin(phi) * math.cos(theta)
    y = fixed_distance * math.sin(phi) * math.sin(theta)
    z = fixed_distance * math.cos(phi)

    # Set camera location
    bpy.context.scene.camera.location = (x, y, z)

    # Point the camera towards the object
    look_at(obj)
    
def apply_random_material(obj):
    # Create a new material
    mat = bpy.data.materials.new(name="Random_Material")

    # Enable 'Use nodes'
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Get the Principled BSDF (the default shader node)
    principled_bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')

    # Randomize color
    principled_bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    
    # Randomize other properties (e.g., roughness, metallic)
    principled_bsdf.inputs['Roughness'].default_value = random.random()
    principled_bsdf.inputs['Metallic'].default_value = random.random()

    # Assign it to the object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)




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
fixed_distance = 3  # Fixed distance from the object, adjust as needed 
target_max_dimension = 1

for stl_path in stl_file_paths:
    clear_scene()
    setup_lights()
    load_stl(stl_path)

    obj = bpy.context.selected_objects[0]
    resize_object(obj, target_max_dimension)  # Resize the object
    set_origin_to_center_of_geometry(obj) 

    camera = setup_camera()
    position_camera(obj, fixed_distance)  # Position the camera

    for i in range(num_images_per_object):
        apply_random_material(obj)  # Apply random material for each image
        angle_x = random.randint(0, 360)
        angle_y = random.randint(0, 360)
        position_camera(obj, fixed_distance)  # Position the camera
        look_at(obj)  # Ensure the camera is looking at the object
        render_save_image(obj.name, angle_x, angle_y, output_directory, i)
        print('rendered')


print("Dataset generation complete.")

