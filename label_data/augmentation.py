import Augmentor

p = Augmentor.Pipeline("./DaTi_fin_img")
p.ground_truth("./DaTi_fin_mask")
p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)
p.sample(50)
