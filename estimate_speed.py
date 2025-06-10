def estimate_speed(pixels_moved, fps, meters_per_pixel):
    meters_per_frame = pixels_moved * meters_per_pixel
    meters_per_second = meters_per_frame * fps
    km_per_hour = meters_per_second * 3.6
    return km_per_hour