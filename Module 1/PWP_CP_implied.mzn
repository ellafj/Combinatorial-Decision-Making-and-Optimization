% Constraint program for present wrapping problem (PWP) with implied constraints
include "globals.mzn";

% Declaring integers
int: w;            % width of paper
int: h;            % Height of paper
int: nPres;        % number of presents to be wrapped
%int: area = w*h;   % area of paper

% Defining sets
set of int: WIDTH = 1..w;
set of int: HEIGHT = 1..h;
set of int: PRESENTS = 1..nPres;
set of int: VALS = 2..nPres;
set of int: DIMS = 1..nPres;
%set of int: EXTRA = 0..area;

% Declaring arrays/grids
array[PRESENTS,1..2] of HEIGHT: dims;     % the dimensions of the presents
array[PRESENTS,1..2] of var 1..h: place;  % the optimal placement of presents

% Declaring constraints
%constraint forall(i in VALS)(place[i-1,1] + dims[i-1,1] <= place[i,1]);
%constraint forall(i in VALS)(place[i-1,2] + dims[i-1,2] <= place[i,2]);

constraint forall(i, j in PRESENTS)(if i < j then place[i,1] + dims[i,1] <= place[j,1] endif);
constraint forall(i, j in PRESENTS)(if i < j then place[i,2] + dims[i,2] <= place[j,2] endif);

constraint forall(i in PRESENTS)(place[i,1] + dims[i,1] <= w /\ place[i,1] >= 0);
constraint forall(i in PRESENTS)(place[i,2] + dims[i,2] <= h /\ place[i,2] >= 0);
constraint sum(i in PRESENTS)(place[i,1]) < w;
constraint sum(i in PRESENTS)(place[i,2]) < h;

% Place presents on paper?


ann: search_ann;
search_ann = int_search(place, input_order, indomain_min, complete); 
%search_ann = int_search(place, input_order, indomain_median, complete); 
%search_ann = int_search(place, first_fail, indomain_min, complete); 
%search_ann = int_search(place, first_fail, indomain_median, complete);

solve :: search_ann satisfy;

output ["place: \(place)\n"];
output ["dims: \(dims)\n"];




