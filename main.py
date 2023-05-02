import os
from multiprocessing import Pool

from task_one_processing import process_one_game
from bonus_task_processing import process_one_image

def main():
    submission_dir_path = "506_Ariton_Cosmin"
    input_dir_path = "train"

    regular_tasks(submission_dir_path, input_dir_path, number_or_workers=5, visualize=False)

    print("\n ##################BONUS_TASK######################### \n")

    bonus_task(submission_dir_path, input_dir_path, number_or_workers=5, visualize=False)
    

def regular_tasks(submission_dir_path, input_dir_path, number_or_workers=1, visualize=False):

    input_dir_path = os.path.join(input_dir_path, "regular_tasks")

    if not os.path.exists(submission_dir_path):
        os.mkdir(submission_dir_path)

    OUTPUT_DIR = os.path.join(submission_dir_path, "regular_tasks")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    moves_filepaths = [os.path.join(input_dir_path, filename) for filename in os.listdir(input_dir_path) if "moves" in filename]
    
    game_info = list()
    for move_file in moves_filepaths:
        with open(move_file, "r") as f:
            file = f.read()

            file = file.split("\n")
            image_paths = [os.path.join(input_dir_path, line.split(" ")[0]) for line in file if line != ""]
            player_turns = [line.split(" ")[1] for line in file if line != ""]

            game_info.append((image_paths, player_turns))

    with Pool(number_or_workers) as p:
        p.starmap(process_one_game, zip([OUTPUT_DIR for _ in game_info], game_info, [visualize for _ in game_info]))

    #list(map(process_one_game, [OUTPUT_DIR for _ in game_info], game_info, [visualize for _ in game_info]))



def bonus_task(submission_dir_path, input_dir_path, number_or_workers=1, visualize=False):
    input_dir_path = os.path.join(input_dir_path, "bonus_task")

    if not os.path.exists(submission_dir_path):
        os.mkdir(submission_dir_path)

    OUTPUT_DIR = os.path.join(submission_dir_path, "bonus_task")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    image_paths = [os.path.join(input_dir_path, filepath) for filepath in os.listdir(input_dir_path) if "jpg" in filepath]

    with Pool(number_or_workers) as p:
        p.starmap(process_one_image, zip([OUTPUT_DIR for _ in image_paths], image_paths, [visualize for _ in image_paths]))

    #list(map(process_one_image, [OUTPUT_DIR for _ in image_paths], image_paths, [visualize for _ in image_paths]))


if __name__ == "__main__":
    main()