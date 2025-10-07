import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RECIPES = {
    "oatmeal": {
        "ingredients_file": DATA_DIR / "oatmeal_pp_ingredients.json",
        "instructions_file": DATA_DIR / "oatmeal_pp_instructions.json",
        "output_file": DATA_DIR / "oatmeal_pp_combined.json",
        "metadata": {
            "recipe_name": "easy-oatmeal-chocolate-chip-cookies",
            "food_role": ["Dessert"],
            "data_provenance": {
                "source_url": "https://www.instructables.com/Easy-Oatmeal-Chocolate-Chip-Cookies/",
                "last_system_access": datetime(2024, 10, 5, 10, 0, 0).isoformat(sep=" ")
            },
            "macronutrients": {
                "serving_size": "1 cookie",
                "calories": "120",
                "protein_g": "2",
                "fat_g": "6",
                "carbohydrates_g": "16"
            }
        },
        "flags": {
            "hasDairy": True,
            "hasMeat": False,
            "hasNuts": False
        }
    },
    "blueberry": {
        "ingredients_file": DATA_DIR / "blueberry_pp_ingredients.json",
        "instructions_file": DATA_DIR / "blueberry_pp_instructions.json",
        "output_file": DATA_DIR / "blueberry_pp_combined.json",
        "metadata": {
            "recipe_name": "ricotta-blueberry-cake-with-streusel-crumble",
            "food_role": ["Dessert"],
            "data_provenance": {
                "source_url": "https://www.instructables.com/Ricotta-Blueberry-Cake-With-Streusel-Crumble/",
                "last_system_access": datetime(2024, 10, 5, 10, 5, 0).isoformat(sep=" ")
            },
            "macronutrients": {
                "serving_size": "1 slice",
                "calories": "320",
                "protein_g": "6",
                "fat_g": "15",
                "carbohydrates_g": "42"
            }
        },
        "flags": {
            "hasDairy": True,
            "hasMeat": False,
            "hasNuts": False
        }
    }
}


def combine_partial(recipe_key: str) -> Path:
    if recipe_key not in RECIPES:
        raise KeyError(f"Unknown recipe key: {recipe_key}")

    config = RECIPES[recipe_key]
    with open(config["ingredients_file"], "r", encoding="utf-8") as f:
        ingredient_payload = json.load(f)
    with open(config["instructions_file"], "r", encoding="utf-8") as f:
        instruction_payload = json.load(f)

    combined = {
        "recipe_name": config["metadata"]["recipe_name"],
        "food_role": config["metadata"].get("food_role", []),
        "data_provenance": config["metadata"].get("data_provenance", {}),
        "macronutrients": config["metadata"].get("macronutrients", {}),
        "ingredients": ingredient_payload.get("ingredients", []),
        "hasDairy": config["flags"].get("hasDairy"),
        "hasMeat": config["flags"].get("hasMeat"),
        "hasNuts": config["flags"].get("hasNuts"),
        "prep_time": instruction_payload.get("prep_time", ""),
        "cook_time": instruction_payload.get("cook_time", ""),
        "serves": instruction_payload.get("serves", ""),
        "instructions": instruction_payload.get("instructions", [])
    }

    output_path = config["output_file"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return output_path


def main() -> None:
    for key in RECIPES:
        output = combine_partial(key)
        print(f"Combined partial outputs for '{key}' -> {output.relative_to(DATA_DIR)}")


if __name__ == "__main__":
    main()
