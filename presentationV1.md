# Presentation 1

## 1. The dataset
The dataset that we have used consist of 1 million final positions of completed 7x7 hex games. Each row consist of a position and the winner. Example:
```
[-1,1,0,0,-1,-1,1,0,1,1,-1,-1,1,1,0,1,1,-1,1,1,0,1,1,-1,1,0,-1,0,-1,0,1,-1,1,0,-1,-1,1,0,-1,1,-1,-1,1,-1,0,1,-1,0,-1,1]
```
The first 49 entries represent what piece occupy the cells on the board. The 50th entry contains the winning side. 1 for black and -1 for white.
## 2. Representing the dataset as a 2d array (matrix)
The dataset can be represented as a 2-dimensional array whereas entries 0 to 6 contain the first row, 7-13 contain the 2nd row and so on.

```
[[-1 -1 -1  0  0 -1  1]
 [ 0  1  1 -1  1 -1 -1]
 [ 1 -1  1 -1 -1 -1  1]
 [-1  1  1  0 -1 -1  1]
 [-1 -1  1 -1  1  1  1]
 [ 1  1  0  1  0  0  1]
 [ 1  1  1  0 -1 -1 -1]]
```
## 3. Booleanizing the matrix
As we can see, each entry in the matrix can contain values -1, 0 and 1, so how can we booleanize it? A boolean can only be False or True, 0 or 1. One way to do that is to expand the matrix with additional values. (Idea from Playing the game of Hex with the
Tsetlin Machine and tree search
Audun Linjord Simonsen
Ole Andr´e Haddelands masters thesis). So our 7x7 matrix now becomes a 7x14 matrix:

```
[[1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0.]
 [0. 1. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1.]
 [1. 0. 0. 0. 1. 1. 0. 0. 1. 1. 0. 0. 0. 1.]
 [1. 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 1. 1. 1.]
 [0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 0. 1.]
 [0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0.]]
```

## 4. Using the graph tsetlin machine to learn
Now that we have booleanized the position, we can use the graph tsetlin machine to try and learn something from it. We modified the example in MNISTConvolutionDemo.py to make it work. (Show [code](./hexgameV1.py) and demo)