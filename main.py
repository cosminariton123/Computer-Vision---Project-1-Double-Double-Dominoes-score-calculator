import os
from multiprocessing import Pool
from tqdm import tqdm

from task_one_processing import process_one_game

def main():
    submission_dir_path = "506_Ariton_Cosmin"
    input_dir_path = "train"

    regular_tasks(submission_dir_path, input_dir_path, number_or_workers=1, visualize=False)
    

def regular_tasks(submission_dir_path, input_dir_path, number_or_workers=1, visualize=False):

    input_dir_path = os.path.join(input_dir_path, "regular_tasks")

    if not os.path.exists(submission_dir_path):
        os.mkdir(submission_dir_path)

    OUTPUT_DIR = os.path.join(submission_dir_path, "regular_task")
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
        results = p.starmap(process_one_game, zip(game_info, [visualize for _ in game_info]))

    #results = list(map(process_one_game, game_info, [True for _ in game_info]))

    print(results)


def bonus_task():
    pass


if __name__ == "__main__":
    main()