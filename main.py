import os
from multiprocessing import Pool
from tqdm import tqdm

from task_one_processing import process_one_regular_task_image

def main():
    submission_dir_path = "506_Ariton_Cosmin"
    input_dir_path = os.path.join("train", "regular_tasks")

    regular_tasks(submission_dir_path, input_dir_path, number_or_workers=5, visualize=True)
    

def regular_tasks(submission_dir_path, input_dir_path, number_or_workers=1, visualize=False):

    if not os.path.exists(submission_dir_path):
        os.mkdir(submission_dir_path)

    OUTPUT_DIR = os.path.join(submission_dir_path, "regular_task")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    input_dir_path = input_dir_path

    image_names = [filename for filename in os.listdir(input_dir_path) if filename.split(".")[-1] == "jpg"]
    image_paths = [os.path.join(input_dir_path, filename) for filename in image_names]

    #with Pool(number_or_workers) as p:
    #    results = p.starmap(process_one_regular_task_image, zip(image_paths, [False for _ in image_paths]))

    results = list(map(process_one_regular_task_image, image_paths, [False for _ in image_paths]))

    #print(results)


def bonus_task():
    pass


if __name__ == "__main__":
    main()