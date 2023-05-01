1. the libraries required to run the project including the full version of each library

python 3.10.10 was used
All the libraries required are in "requirement.txt" file. They are not here to prevent inconsitencies!

2. how to run your solution and where to look for the output file.

script: main.py
"board+dominoes" and "test" directories are expected to exist. They should be the same format as the ones provided. Furthermore, boards+dominoes should have the same images as the ones provided.

function main() to run regular_tasks() and bonus_task()
function regular_tasks() to run all regular tasks solvers.
function bonus_task() to run the bonus task solver

Output will be "506_Ariton_Cosmin" in the root directory. Subdirectory regular_tasks will contain the outputs of the regular_tasks and subdirectory bonus_task will contain the output of bonus task. Obviously, the directories will appear only if the corresponding functions will be ran. 

script: solve_problem.py
function: solve_problem(input_folder_name), where input_folder_name is the path to the folder containing the test images
output: the output file consists of txt files in results/...