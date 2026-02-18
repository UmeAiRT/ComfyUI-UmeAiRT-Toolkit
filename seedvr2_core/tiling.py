"""Tiling utilities for dividing images into overlapping tiles."""


def calculate_efficient_tile_size(width, height):
    """Calculate GPU-efficient tile dimensions by padding to optimal sizes."""
    min_efficient_size = 512

    efficient_width = max(min_efficient_size, ((width + 15) // 16) * 16)
    efficient_height = max(min_efficient_size, ((height + 15) // 16) * 16)

    return efficient_width, efficient_height


def generate_tiles(image, tile_width, tile_height, padding, strategy):
    """Generate tiles based on the specified strategy."""
    width, height = image.size
    tiles = []

    if strategy == "Linear":
        for y in range(0, height, tile_height):
            for x in range(0, width, tile_width):
                tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
    elif strategy == "Chess":
        for y_idx, y in enumerate(range(0, height, tile_height)):
            for x_idx, x in enumerate(range(0, width, tile_width)):
                if (x_idx + y_idx) % 2 == 0:
                    tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
        for y_idx, y in enumerate(range(0, height, tile_height)):
            for x_idx, x in enumerate(range(0, width, tile_width)):
                if (x_idx + y_idx) % 2 != 0:
                    tiles.append(get_tile_info(image, x, y, tile_width, tile_height, padding))
    return tiles


def get_tile_info(image, x, y, tile_width, tile_height, padding):
    """Extract tile information and crop the tile with padding.

    Uses edge extension (reflection) for memory padding instead of solid color fill
    to avoid artificial edges that the AI upscaler would process as real content.
    """
    from PIL import Image, ImageOps
    import numpy as np

    width, height = image.size

    # Calculate actual tile boundaries (may be smaller at edges)
    actual_tile_width = min(tile_width, width - x)
    actual_tile_height = min(tile_height, height - y)

    # Calculate padding (only add padding where there are adjacent tiles)
    left_pad = padding if x > 0 else 0
    top_pad = padding if y > 0 else 0
    right_pad = padding if x + tile_width < width else 0
    bottom_pad = padding if y + tile_height < height else 0

    # Create the padded crop box
    padded_box = (
        max(0, x - left_pad),
        max(0, y - top_pad),
        min(width, x + actual_tile_width + right_pad),
        min(height, y + actual_tile_height + bottom_pad),
    )

    tile = image.crop(padded_box)

    # Calculate efficient dimensions for GPU processing
    current_width, current_height = tile.size
    efficient_width, efficient_height = calculate_efficient_tile_size(current_width, current_height)

    # Add memory padding if needed for GPU efficiency
    memory_pad_right = efficient_width - current_width
    memory_pad_bottom = efficient_height - current_height

    if memory_pad_right > 0 or memory_pad_bottom > 0:
        tile_array = np.array(tile.crop((0, 0, current_width, current_height)))

        # Create output array
        padded_array = np.zeros((efficient_height, efficient_width, 3), dtype=np.uint8)

        # Copy original tile
        padded_array[:current_height, :current_width] = tile_array

        # Reflect right edge
        if memory_pad_right > 0:
            reflect_width = min(memory_pad_right, current_width)
            padded_array[:current_height, current_width:current_width + reflect_width] = \
                tile_array[:, current_width - reflect_width:current_width][:, ::-1]
            if memory_pad_right > reflect_width:
                edge_col = tile_array[:, -1:].repeat(memory_pad_right - reflect_width, axis=1)
                padded_array[:current_height, current_width + reflect_width:] = edge_col

        # Reflect bottom edge
        if memory_pad_bottom > 0:
            reflect_height = min(memory_pad_bottom, current_height)
            padded_array[current_height:current_height + reflect_height, :current_width] = \
                tile_array[current_height - reflect_height:current_height, :][::-1, :]
            if memory_pad_bottom > reflect_height:
                edge_row = tile_array[-1:, :].repeat(memory_pad_bottom - reflect_height, axis=0)
                padded_array[current_height + reflect_height:, :current_width] = edge_row

        # Handle corner (bottom-right)
        if memory_pad_right > 0 and memory_pad_bottom > 0:
            reflect_width = min(memory_pad_right, current_width)
            reflect_height = min(memory_pad_bottom, current_height)
            corner = tile_array[current_height - reflect_height:current_height,
                               current_width - reflect_width:current_width]
            padded_array[current_height:current_height + reflect_height,
                        current_width:current_width + reflect_width] = corner[::-1, ::-1]

            if memory_pad_right > reflect_width:
                padded_array[current_height:current_height + reflect_height,
                            current_width + reflect_width:] = \
                    padded_array[current_height:current_height + reflect_height,
                                current_width + reflect_width - 1:current_width + reflect_width].repeat(
                                    memory_pad_right - reflect_width, axis=1)
            if memory_pad_bottom > reflect_height:
                padded_array[current_height + reflect_height:,
                            current_width:current_width + reflect_width] = \
                    padded_array[current_height + reflect_height - 1:current_height + reflect_height,
                                current_width:current_width + reflect_width].repeat(
                                    memory_pad_bottom - reflect_height, axis=0)
            if memory_pad_right > reflect_width and memory_pad_bottom > reflect_height:
                padded_array[current_height + reflect_height:,
                            current_width + reflect_width:] = tile_array[-1, -1]

        tile = Image.fromarray(padded_array)

    return {
        "tile": tile,
        "position": (x, y),
        "actual_size": (actual_tile_width, actual_tile_height),
        "padding": (left_pad, top_pad, right_pad, bottom_pad),
        "memory_padding": (0, 0, memory_pad_right, memory_pad_bottom),
        "original_tile_size": (current_width, current_height),
    }
