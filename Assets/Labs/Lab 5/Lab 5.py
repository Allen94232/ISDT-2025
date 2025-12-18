import openai
import os
import json
import base64
from pathlib import Path
from pydantic import BaseModel

# Optional: graphviz for visualization (requires system installation)
try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️  Graphviz not available. Visualization will be skipped.")

# ==============================
# Configuration
# ==============================
API_KEY = os.getenv("OPENAI_API_KEY")

# three goals
goals = [
    "make the plant grow",
    "clean the dirty surface",
    "keep yourself dry in the rain"
]

# first set of objects
object_set_1 = ["bucket", "hammer", "coffee cup", "alcohol spray", "umbrella"]

# second set of objects
object_set_2 = ["towel", "fan", "flashlight", "water bottle", "notebook"]

# ==============================
# OpenAI Initialization
# ==============================
def initialize_openai():
    openai.api_key = API_KEY
    print("API Key set successfully.")
    return openai.OpenAI()

# ==============================
# Vision API Functions
# ==============================
def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def identify_objects_from_image(image_path, client):
    """Use GPT-4 Vision to identify the main object in an image (single object per image)"""
    print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Call GPT-4 Vision API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at identifying the main object in images.\n"
                    "Each image contains ONE main object.\n"
                    "Identify the primary object and return ONLY its name.\n"
                    "Use simple, common names (e.g., 'cup' not 'ceramic mug').\n"
                    "Return ONLY ONE object name, nothing else.\n"
                    "Example: bucket"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the main object in this image? Give me only the object name."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=50,
        temperature=0.2
    )
    
    # Parse response - take only the first object mentioned
    objects_text = response.choices[0].message.content.strip()
    
    # Handle multiple objects if GPT returns more than one (take first only)
    if ',' in objects_text:
        main_object = objects_text.split(',')[0].strip()
        print(f"   ⚠️  Multiple objects detected, using first one: {main_object}")
    else:
        main_object = objects_text
        print(f"   ✓ Identified object: {main_object}")
    
    return main_object

def get_objects_from_images(images_dir="images", client=None):
    """Load images from directory and identify ONE object per image using GPT-4 Vision"""
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"❌ Error: '{images_dir}' folder not found!")
        print(f"   Please create the folder and add images.")
        return []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ Error: No images found in '{images_dir}' folder!")
        return []
    
    print(f"\n📸 Found {len(image_files)} images in '{images_dir}' folder")
    print("="*60)
    
    object_list = []
    
    for img_file in sorted(image_files)[:5]:  # Process up to 5 images
        obj = identify_objects_from_image(str(img_file), client)
        object_list.append(obj)
    
    print("="*60)
    print(f"✅ Identified {len(object_list)} objects (one per image)")
    print(f"   Objects: {', '.join(object_list)}\n")
    
    return object_list

# ==============================
# Pydantic data models
# ==============================
class PuzzleEdge(BaseModel):
    source: str
    relation: str
    target: str

class PuzzleGraph(BaseModel):
    goal: str
    objects: list[str]
    edges: list[PuzzleEdge]

# ==============================
# Call API
# ==============================
def call_openai(goal, objects, response_format, client):
    prompt = f"""
Goal: {goal}
Objects: {', '.join(objects)}

Generate a puzzle graph describing how these objects interact to achieve the goal.
Each edge should describe an action or state change (e.g., 'umbrella PLACE ON seed' or 'seed: dry → moist').
Output must use all objects at least once.
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a creative assistant that generates puzzle graphs.\n"
                    "Each puzzle graph describes how a set of real-world objects interact "
                    "to achieve a given goal.\n"
                    "Use all objects in the list at least once.\n"
                    "Each edge should represent an action or state change, like "
                    "'object PLACE ON seed' or 'seed: dry → moist'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        response_format=response_format,
    )
    return completion.choices[0].message.parsed

# ==============================
# Visualization Function
# ==============================
def visualize_puzzle_graphs(json_path="puzzle_graphs.json", output_dir="puzzle_graphs_viz"):
    """Visualize puzzle graphs using Graphviz (requires system Graphviz installation)"""
    if not GRAPHVIZ_AVAILABLE:
        print("\n⚠️  Graphviz visualization skipped (graphviz not installed)")
        print("   To enable visualization:")
        print("   1. Install Graphviz: winget install graphviz")
        print("   2. Restart your terminal")
        print("   3. Run the script again")
        return
    
    try:
        os.makedirs(output_dir, exist_ok=True)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, puzzle in enumerate(data):
            goal = puzzle["goal"]
            edges = puzzle["edges"]

            # set up graph
            dot = Digraph(comment=goal)
            dot.attr(rankdir="LR")  # 左→右排列（可改成 "TB" 直式）
            dot.attr("node", shape="ellipse", style="filled", color="lightgrey")
            
            # Add goal node with special styling
            safe_goal = goal.replace(" ", "_").replace("/", "_")
            dot.node("GOAL", label=goal, shape="box", style="filled", color="lightblue", fontsize="14", fontweight="bold")

            # Track all nodes that appear
            all_nodes = set()
            for edge in edges:
                all_nodes.add(edge["source"])
                all_nodes.add(edge["target"])

            # add edges from puzzle
            for edge in edges:
                src = edge["source"]
                rel = edge["relation"]
                tgt = edge["target"]
                dot.edge(src, tgt, label=rel)

            # Find terminal nodes (nodes that don't appear as source in any edge)
            source_nodes = {edge["source"] for edge in edges}
            target_nodes = {edge["target"] for edge in edges}
            terminal_nodes = target_nodes - source_nodes
            
            # Connect all terminal nodes to GOAL
            for terminal in terminal_nodes:
                dot.edge(terminal, "GOAL", label="achieves", color="green", fontcolor="green", style="bold")

            # render graph to file
            output_path = os.path.join(output_dir, f"{i+1}_{safe_goal}")
            dot.render(output_path, format="png", cleanup=True)

            print(f"✅ Saved: {output_path}.png (connected to goal: {goal})")

        print("\n🎨 All puzzle graphs visualized successfully!")
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        print("   Make sure Graphviz is installed: winget install graphviz")

# ==============================
# Main
# ==============================
def main(use_vision=True):
    client = initialize_openai()
    response_format = PuzzleGraph
    results = []

    # Prepare object sets: Set 1 = hardcoded, Set 2 = from images
    object_sets = []
    
    # Set 1: Always use hardcoded objects
    print("\n🎯 Object Set 1: Using hardcoded objects")
    print(f"   Objects: {', '.join(object_set_1)}")
    object_sets.append(object_set_1)
    
    # Set 2: Get objects from images (one object per image)
    if use_vision:
        print("\n🎯 Object Set 2: Using GPT-4 Vision to identify objects from images...")
        vision_objects = get_objects_from_images("images", client)
        
        if vision_objects:
            # vision_objects is now a list of strings (one object per image)
            object_sets.append(vision_objects)
            print(f"   Objects from images: {', '.join(vision_objects)}")
        else:
            print("⚠️  No objects found from images, using hardcoded Set 2 instead.")
            object_sets.append(object_set_2)
    else:
        print("\n🎯 Object Set 2: Using hardcoded objects (vision disabled)")
        object_sets.append(object_set_2)
    
    # Generate puzzle graphs for each object set
    for idx, objects in enumerate(object_sets):
        print(f"\n{'='*60}")
        print(f"=== Puzzle Graphs for Object Set {idx + 1} ===")
        print(f"Objects: {', '.join(objects)}")
        print(f"{'='*60}")
        
        for goal in goals:
            response = call_openai(goal, objects, response_format, client)
            print(f"\nGoal: {goal}")
            print(response.model_dump_json(indent=2))
            results.append(response.model_dump())

    # Remove colons from all string values before saving
    def remove_colons(obj):
        """Recursively remove colons from all strings in a data structure"""
        if isinstance(obj, dict):
            return {k: remove_colons(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_colons(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace(":", " ")  # Replace colon with space
        else:
            return obj
    
    # Apply colon removal to all results
    cleaned_results = remove_colons(results)
    
    # output all results to a JSON file
    with open("puzzle_graphs.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    print("\n✅ All puzzle graphs saved to puzzle_graphs.json (colons removed)")
    print(f"   Total graphs generated: {len(results)}")
    print(f"   Location: {os.path.abspath('puzzle_graphs.json')}")

if __name__ == "__main__":
    # Set use_vision=True to use GPT-4 Vision API with images
    # Set use_vision=False to use hardcoded object sets
    main(use_vision=True)
    
    # Try to visualize (will skip if Graphviz not installed)
    visualize_puzzle_graphs("puzzle_graphs.json")

