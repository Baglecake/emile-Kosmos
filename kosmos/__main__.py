"""Entry point for emile-Kosmos: python -m kosmos [--model MODEL] [--size SIZE] [--speed SPEED]"""

import sys
from .world.grid import KosmosWorld
from .agent.core import KosmosAgent
from .render.pygame_render import KosmosRenderer
from .persistence import save_state, load_state


def main():
    # Parse args
    model = "llama3.1:8b"
    size = 30
    speed = 8
    seed = None
    save_path = None
    load_path = None
    autosave = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--size" and i + 1 < len(args):
            size = int(args[i + 1])
            i += 2
        elif args[i] == "--speed" and i + 1 < len(args):
            speed = int(args[i + 1])
            i += 2
        elif args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] == "--save" and i + 1 < len(args):
            save_path = args[i + 1]
            i += 2
        elif args[i] == "--load" and i + 1 < len(args):
            load_path = args[i + 1]
            i += 2
        elif args[i] == "--no-autosave":
            autosave = False
            i += 1
        elif not args[i].startswith("--"):
            model = args[i]
            i += 1
        else:
            i += 1

    print(f"  emile-Kosmos")
    print(f"  World: {size}x{size}  Model: {model}  Speed: {speed}/s")

    world = KosmosWorld(size=size, seed=seed)
    agent = KosmosAgent(world, model=model)

    if load_path:
        try:
            load_state(load_path, world, agent)
            print(f"  Loaded state from {load_path}")
        except Exception as e:
            print(f"  Failed to load state: {e}")

    renderer = KosmosRenderer(world, agent, cell_size=max(12, 600 // size))
    renderer.speed = speed
    renderer.run()

    # Auto-save on quit
    if autosave and save_path:
        try:
            save_state(world, agent, save_path)
            print(f"  State saved to {save_path}")
        except Exception as e:
            print(f"  Failed to save state: {e}")

    print("  Kosmos ended.")


if __name__ == "__main__":
    main()
