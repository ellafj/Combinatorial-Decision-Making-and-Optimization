% Constraint program for present wrapping problem (PWP)
include "globals.mzn";

% Declaring integers
int: w;   % width of paper
int: h;   % Height of paper
int: nPres;  % number of presents to be wrapped

% Defining sets
set of int: WIDTH = 1..w;
set of int: HEIGHT = 1..h;
set of int: PRESENTS = 1..nPres;
set of int: VALS = 1..nPres-1;
set of int: DIMS = 1..nPres;

output ["HEIGHT: \(HEIGHT)"];

% Declaring arrays/grids
array[PRESENTS,1..2] of HEIGHT: dims;     % the dimensions of the presents
array[PRESENTS,1..2] of var 1..h: place;  % the optimal placement of presents

% Declaring constraints
constraint forall(i in VALS)(place[i,1] < place[i+1,1]);
constraint forall(i in VALS)(place[i,2] < place[i+1,2]);
constraint sum(i in VALS)(place[i,1]) < w;






