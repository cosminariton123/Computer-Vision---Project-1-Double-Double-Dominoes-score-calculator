1. the libraries required to run the project including the full version of each library

python 3.10.10 was used
All the libraries required are in "requirement.txt" file. They are not here to prevent inconsitencies!

2. how to run your solution and where to look for the output file.

script: main.py
train and test data present at https://tinyurl.com/CV-2023-Project1
"board+dominoes" directory is expected to exist.
Furthermore, "boards+dominoes" should have the same images as the ones provided.
All files should be the same format as the ones provided for training(including file extentions).
"input_dir_path" variable will be changed as needed to point to the corect input directory. It will most likely be "test" and it is expected to have the same structure as "train".
the "regular_tasks" directory inside "test" is expected to contain x files "x_moves.txt". The images will be processed in the order given in this file. Every player in this file is counted(it doesn't matter if they are 2 or N). It should be noted that "player1" is different than "plaYer1" or other variants. The file shouldn't have mistakes.
the "bonus_task" directory inside "test" is expected to contain ".jpg" images. All the ".jpg" files in this directory will be processed.

function main() to run regular_tasks() and bonus_task()
function regular_tasks() to run all regular tasks solvers.
function bonus_task() to run the bonus task solver

For the sake of time, I used multiprocessing to enable a faster execution. If this is unwanted for some reason, set "number_of_workers" to 1 or follow the instruction in the comments in "main.py" 
"number_of_workers" should be lower or equal than number of games in regular_tasks

Output will be "506_Ariton_Cosmin" in the root directory. Subdirectory regular_tasks will contain the outputs of the regular_tasks and subdirectory bonus_task will contain the output of bonus task. Obviously, the directories will appear only if the corresponding functions will be run. 
The output files will be simmillar to the ones in "evaluation" or "train"

I must also state that I consider "evaluation.py" to be incorectly developed, as:
For Regular tasks:
-it only evaluates games 1 to 4  range(1, 5) => [1, 2, 3, 4], not [1, 2, 3, 4, 5]
-"Points values" is always "0.015", so even if I provide incorect values, I get maximum points

For Bonus task:
It outputs "encountered an error" which indicates a problem with my output format.
This output appears when there is a difference in the number of points enumerated in the file.
Even if my algorithm fails and outputs the wrong number of points, the format is corect.
So, I only expect to be penalized by giving the wrong answer, not by not following the format mandated.