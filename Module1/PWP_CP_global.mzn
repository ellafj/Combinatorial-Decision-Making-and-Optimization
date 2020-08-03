% Constraint program for present wrapping problem (PWP) using global constraints
include "globals.mzn";

% Declaring integers
int: w;            % width of paper
int: h;            % Height of paper
int: nPres;        % number of presents to be wrapped
int: area = w*h;   % area of paper

% Defining sets
set of int: WIDTH = 1..w;
set of int: HEIGHT = 1..h;
set of int: PRESENTS = 1..nPres;
set of int: DIMS = 1..nPres;

% Declaring arrays/grids
array[PRESENTS,1..2] of HEIGHT: dims;     % the dimensions of the presents
array[PRESENTS,1..2] of var 1..h: place;  % the optimal placement of presents

% Declaring constraints
constraint diffn(place[PRESENTS,1], place[PRESENTS,2], dims[PRESENTS,1], dims[PRESENTS,2]);

% Declaring search heuristic
ann: search_ann;
%search_ann = int_search(place, input_order, indomain_min, complete); 
%search_ann = int_search(place, input_order, indomain_median, complete); 
search_ann = int_search(place, first_fail, indomain_min, complete); 
%search_ann = int_search(place, first_fail, indomain_median, complete);

% Solving optimization problem
solve :: search_ann satisfy;

output ["\(place)"];




