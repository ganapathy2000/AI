# AI
**1.Write a program to implement breadth first search using python.<br>**
graph = {<br>
'1' : ['2','10'],<br>
'2' : ['3','8'],<br>
'3' : ['4'],<br>
'4' : ['5','6','7'],<br>
'5' : [],<br>
'6' : [],<br>
'7' : [],<br>
'8' : ['9'],<br>
'9' : [],<br>
'10' : []<br>
}<br>
visited = []<br>
queue = []<br>
def bfs(visited, graph, node):<br>
visited.append(node)<br>
queue.append(node)<br>
while queue:<br>
m = queue.pop(0)<br>
print (m, end = " ")<br>
for neighbour in graph[m]:<br>
if neighbour not in visited:<br>
visited.append(neighbour)<br>
queue.append(neighbour)<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')<br>
Output:-<br>
Following is the Breadth-First Search<br>
1 2 10 3 8 4 9 5 6 7<br>
***********************************************************************************************************************************************************************

**2.write a program to implement a deapth first search using python**.
graph = {<br>
'5' : ['3','7'],<br>
'3' : ['2', '4'],<br>
'7' : ['6'],<br>
'6': [],<br>
'2' : ['1'],<br>
'1':[],<br>
'4' : ['8'],<br>
'8' : []<br>
}<br>
visited = set()<br>
def dfs(visited, graph, node):<br>
if node not in visited:<br>
print (node)<br>
visited.add(node)<br>
for neighbour in graph[node]:<br>
dfs(visited, graph, neighbour)<br>
print("Following is the Depth-First Search")<br>
dfs(visited, graph, '5')<br>
Output:-<br>
Following is the Depth-First Search<br>
5
3
2
1
4
8
7
6
<br>
***********************************************************************************************************************************************************************

**3.write a program to implement water jug problem using python.<br>**
from collections import defaultdict<br>
jug1, jug2, aim = 4, 3, 2<br>
visited = defaultdict(lambda: False)<br>
def waterJugSolver(amt1, amt2):<br>
if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):<br>
print(amt1, amt2)<br>
return True<br>
if visited[(amt1, amt2)] == False:<br>
print(amt1, amt2)<br>
visited[(amt1, amt2)] = True<br>
return (waterJugSolver(0, amt2) or<br>
waterJugSolver(amt1, 0) or<br>
waterJugSolver(jug1, amt2) or<br>
waterJugSolver(amt1, jug2) or<br>
waterJugSolver(amt1 + min(amt2, (jug1-amt1)),<br>
amt2 - min(amt2, (jug1-amt1))) or<br>
waterJugSolver(amt1 - min(amt1, (jug2-amt2)),<br>
amt2 + min(amt1, (jug2-amt2))))<br>
else:<br>
return False<br>
print("Steps: ")<br>
waterJugSolver(0, 0)<br>

Output:-<br>
Steps:
0 0
4 0
4 3
0 3
3 0
3 3
4 2
0 2
True<br>
***********************************************************************************************************************************************************************

**4.write a program to implement tower of hannoi using python.<br>**
def TowerOfHanoi(n , source, destination, auxiliary):<br>
if n==1:<br>
print ("Move disk 1 from source",source,"to destination",destination)<br>
return<br>
TowerOfHanoi(n-1, source, auxiliary, destination)<br>
print ("Move disk",n,"from source",source,"to destination",destination)<br>
TowerOfHanoi(n-1, auxiliary, destination, source)<br>
n = 4<br>
TowerOfHanoi(n,'A','B','C')<br>

Output:-
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 3 from source A to destination C<br>
Move disk 1 from source B to destination A<br>
Move disk 2 from source B to destination C<br>
Move disk 1 from source A to destination C<br>
Move disk 4 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
Move disk 2 from source C to destination A<br>
Move disk 1 from source B to destination A<br>
Move disk 3 from source C to destination B<br>
Move disk 1 from source A to destination C<br>
Move disk 2 from source A to destination B<br>
Move disk 1 from source C to destination B<br>
***********************************************************************************************************************************************************************

#5.Write a Program to Implement Best First Search using Python.<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>

*  for implementing BFS | returns path having lowest cost*<br>
def best_first_search(source, target, n):<br>
    visited = [0] *n<br>
    visited[source] = True<br>
    pq = PriorityQueue()<br>
    pq.put((0, source))<br>
    while pq.empty() == False:<br>
        u = pq.get()[1]<br>
        print(u, end=" ") # the path having lowest cost<br>
        if u == target:<br>
            break<br>
        for v, c in graph[u]:<br>
            if visited[v] == False:<br>
                visited[v] = True<br>
                pq.put((c,v))
print()<br>

  for adding edges to graph<br><br>
def addedge(x, y, cost):<br>
    graph[x].append((y, cost))<br>
    graph[y].append((x, cost))<br>
<br>
v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>

e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
    x, y, z = list(map(int, input().split()))<br>
    addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))<br>
print("\nPath: ", end = "")<br>
best_first_search(source, target, v)<br>
**OUTPUT:**

Enter the number of nodes: 4<br>
Enter the number of edges: 5<br>
Enter the edges along with their weights:<br>
0 1 1 <br>
0 2 1 <br>
0 3 2<br>
2 3 2<br>
1 3 3<br>
Enter the Source Node: 2<br>
Enter the Target/Destination Node: 1<br>

Path: 2 0 1 <br>
​<br>
***********************************************************************************************************************************************************************

#6. write a program using Tic-Tac-Toe Program using<br>
# random number in Python<br>
 
# importing all necessary libraries<br>
import numpy as np<br>
import random<br>
from time import sleep<br>
 
Creates an empty board<br>
 
 
def create_board():<br>
    return(np.array([[0, 0, 0],<br>
                     [0, 0, 0],<br>
                     [0, 0, 0]]))<br>
 
Check for empty places on board<br>
 
 
def possibilities(board):<br>
    l = []<br>
 
    for i in range(len(board)):<br>
        for j in range(len(board)):<br>
 
            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>
 
 Select a random place for the player<br>
 
 <br>
def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>
  Checks whether the player has three<br>
 of their marks in a horizontal row<br>
 
 
def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>
 
        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>
 
        if win == True:<br>
            return(win)<br>
    return(win)<br>
 
 Checks whether the player has three<br>
 of their marks in a vertical row<br>
 
 
def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>
 
        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>
 
        if win == True:<br>
            return(win)<br>
    return(win)<br>
 
 Checks whether the player has three<br>
 of their marks in a diagonal row<br>
 
 
def diag_win(board, player):<br>
    win = True<br>
    y = 0<br>
    for x in range(len(board)):<br>
        if board[x, x] != player:<br>
            win = False<br>
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>
 
 Evaluates whether there is<br>
 a winner or a tie<br>
 
 
def evaluate(board):<br>
    winner = 0<br>
 
    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
                col_win(board, player) or<br>
                diag_win(board, player)):<br>
 
            winner = player<br>
 
    if np.all(board != 0) and winner == 0:<br>
        winner = -1<br>
    return winner<br>
 
 Main function to start the game<br>
 
 
def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)<br>
    sleep(2)<br>
 
    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>
 
 
 Driver Code<br>
print("Winner is: " + str(play_game()))<br>


OUTPUT:<br>
[[0 0 0]<br>
 [0 0 0]<br>
 [0 0 0]]<br>
Board after 1 move<br>
[[0 0 0]
 [1 0 0]
 [0 0 0]]<br>
Board after 2 move<br>
[[0 0 0]<br>
 [1 2 0]<br>
 [0 0 0]]<br>
Board after 3 move<br>
[[0 1 0]<br>
 [1 2 0]<br>
 [0 0 0]]<br>
Board after 4 move<br>
[[0 1 0]<br>
 [1 2 0]<br>
 [0 2 0]]<br>
Board after 5 move<br>
[[0 1 0]<br>
 [1 2 0]<br>
 [1 2 0]]<br>
Board after 6 move<br>
[[0 1 0]<br>
 [1 2 0]<br>
 [1 2 2]]<br>
Board after 7 move<br>
[[0 1 1]<br>
 [1 2 0]<br>
 [1 2 2]]<br>
Board after 8 move<br>
[[0 1 1]<br>
 [1 2 2]<br>
 [1 2 2]]<br>
Board after 9 move<br>
[[1 1 1]<br>
 [1 2 2]<br>
 [1 2 2]]<br>
Winner is: 1<br>
**********************************************************************************************************************************************************************

# 7.program to print the path from root
 node to destination node for N*N-1 puzzle
 algorithm using Branch and Bound
 The solution assumes that instance of
 puzzle is solvable

 Importing copy for deepcopy function
import copy

#Importing the heap functions from python
 library for Priority Queue
from heapq import heappush, heappop

 This variable can be changed to change
the program from 8 puzzle(n=3) to 15
 puzzle(n=4) to 24 puzzle(n=5)...
n = 3

 bottom, left, top, right
row = [ 1, 0, -1, 0 ]
col = [ 0, -1, 0, 1 ]

 A class for Priority Queue
class priorityQueue:
	
	 Constructor to initialize a
	 Priority Queue
	def __init__(self):
		self.heap = []

	 Inserts a new key 'k'
	def push(self, k):
		heappush(self.heap, k)

	 Method to remove minimum element
	 from Priority Queue
	def pop(self):
		return heappop(self.heap)

	 Method to know if the Queue is empty
	def empty(self):
		if not self.heap:
			return True
		else:
			return False

# Node structure
class node:
	
	def __init__(self, parent, mat, empty_tile_pos,
				cost, level):
					
		# Stores the parent node of the
		# current node helps in tracing
		# path when the answer is found
		self.parent = parent

		# Stores the matrix
		self.mat = mat

		# Stores the position at which the
		# empty space tile exists in the matrix
		self.empty_tile_pos = empty_tile_pos

		# Storesthe number of misplaced tiles
		self.cost = cost

		# Stores the number of moves so far
		self.level = level

	# This method is defined so that the
	# priority queue is formed based on
	# the cost variable of the objects
	def __lt__(self, nxt):
		return self.cost < nxt.cost

 Function to calculate the number of
 misplaced tiles ie. number of non-blank
 tiles not in their goal position
def calculateCost(mat, final) -> int:
	
	count = 0
	for i in range(n):
		for j in range(n):
			if ((mat[i][j]) and
				(mat[i][j] != final[i][j])):
				count += 1
				
	return count

def newNode(mat, empty_tile_pos, new_empty_tile_pos,
			level, parent, final) -> node:
				
	# Copy data from parent matrix to current matrix
	new_mat = copy.deepcopy(mat)

	# Move tile by 1 position
	x1 = empty_tile_pos[0]
	y1 = empty_tile_pos[1]
	x2 = new_empty_tile_pos[0]
	y2 = new_empty_tile_pos[1]
	new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]

	# Set number of misplaced tiles
	cost = calculateCost(new_mat, final)

	new_node = node(parent, new_mat, new_empty_tile_pos,
					cost, level)
	return new_node

 Function to print the N x N matrix
def printMatrix(mat):
	
	for i in range(n):
		for j in range(n):
			print("%d " % (mat[i][j]), end = " ")
			
		print()

 Function to check if (x, y) is a valid
 matrix coordinate
def isSafe(x, y):
	
	return x >= 0 and x < n and y >= 0 and y < n

 Print path from root node to destination node<br>
def printPath(root):<br>
	
	if root == None:
		return
	
	printPath(root.parent)<br>
	printMatrix(root.mat)<br>
	print()<br>

 Function to solve N*N - 1 puzzle algorithm<br>
using Branch and Bound. empty_tile_pos is<br>
 the blank tile position in the initial state.<br>
def solve(initial, empty_tile_pos, final):<br>
	
	Create a priority queue to store live
	 nodes of search tree
	pq = priorityQueue()

	 Create the root node
	cost = calculateCost(initial, final)
	root = node(None, initial,
				empty_tile_pos, cost, 0)

	 Add root to list of live nodes
	pq.push(root)

	 Finds a live node with least cost,
	 add its children to list of live
	 nodes and finally deletes it from
	 the list.
	while not pq.empty():

		 Find a live node with least estimated
		 cost and delete it form the list of
		 live nodes
		minimum = pq.pop()

		 If minimum is the answer node
		if minimum.cost == 0:
			
			# Print the path from root to
			# destination;
			printPath(minimum)
			return

		 Generate all possible children
		for i in range(n):
			new_tile_pos = [
				minimum.empty_tile_pos[0] + row[i],
				minimum.empty_tile_pos[1] + col[i], ]
				
			if isSafe(new_tile_pos[0], new_tile_pos[1]):
				
				# Create a child node
				child = newNode(minimum.mat,
								minimum.empty_tile_pos,
								new_tile_pos,
								minimum.level + 1,
								minimum, final,)

				# Add child to list of live nodes
				pq.push(child)

 Driver Code

 Initial configuration
 Value 0 is used for empty space
initial = [ [ 1, 2, 3 ],
			[ 5, 6, 0 ],
			[ 7, 8, 4 ] ]

 Solvable Final configuration<br>
 Value 0 is used for empty space<br>
final = [ [ 1, 2, 3 ],
		[ 5, 8, 6 ],
		[ 0, 7, 4 ] ]

 Blank tile coordinates in
 initial configuration
empty_tile_pos = [ 1, 2 ]

 Function call to solve the puzzle
solve(initial, empty_tile_pos, final)<br>

8.Write a Program to Implement Travelling Salesman problem using Python:<br>

from sys import maxsize<br>
from itertools import permutations<br>
V = 4
def travellingSalesmanProblem(graph, s):<br>
    vertex = []<br>
    for i in range(V):
        if i != s:
            vertex.append(i)<br>
    min_path = maxsize<br>
    next_permutation=permutations(vertex)<br>
    for i in next_permutation:<br>
        current_pathweight = 0<br>
        k = s<br>
        for j in i:<br>
            current_pathweight += graph[k][j]<br>
            k = j<br>
        current_pathweight += graph[k][s]<br>
        min_path = min(min_path, current_pathweight)<br>
    return min_path<br>
if __name__ == "__main__":<br>
    graph = [[0, 10, 15, 20], [10, 0, 35, 25],<br>
        [15, 35, 0, 30], [20, 25, 30, 0]]<br>
    s = 0<br>
    print(travellingSalesmanProblem(graph, s))<br>
    
    OUTPUT:
    80<br>
  **********************************************************************************************************************************************************************
 
 8.Program to implement and FIND-S Algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.<br>
    import pandas as pd<br>
import numpy as np<br>
 
#to read the data in the csv file<br>
data = pd.read_csv("Traineg.csv")<br>
print(data)<br>
 
#making an array of all the attributes<br>
d = np.array(data)[:,:-1]<br>
print("The attributes are: ",d)<br>
 
#segragating the target that has positive and negative examples<br>
target = np.array(data)[:,-1]<br>
print("The target is: ",target)<br>
 
#training function to implement find-s algorithm<br><br>
def train(c,t):<br>
    for i, val in enumerate(t):<br>
        if val == "Yes":<br>
            specific_hypothesis = c[i].copy()<br>
            break<br>
             
    for i, val in enumerate(c):<br>
        if t[i] == "Yes":<br>
            for x in range(len(specific_hypothesis)):<br>
                if val[x] != specific_hypothesis[x]:<br>
                    specific_hypothesis[x] = '?'<br>
                else:<br>
                    pass<br>
                return specific_hypothesis<br>
 
#obtaining the final hypothesis<br>
print("The final hypothesis is:",train(d,target))<br>
OUTPUT:<br>
   Sunny   Warm  Normal   Strong  Warm.1     Same   Yes<br>
0  Sunny   Warm    High   Strong    Warm     Same   Yes<br>
1  Rainy   Cold    High   Strong    Warm   Change    No<br>
2  Sunny   Warm    High   Strong    Cool   Change   Yes<br>
The attributes are:  [['Sunny' ' Warm' ' High' ' Strong' ' Warm' ' Same']<br>
 ['Rainy' ' Cold' ' High' ' Strong' ' Warm' ' Change']<br>
 ['Sunny' ' Warm' ' High' ' Strong' ' Cool' ' Change']]<br>
The target is:  [' Yes' ' No' ' Yes']<br>
The final hypothesis is: None<br>
***********************************************************************************************************************************************************************
https://copyassignment.com/diabetes-prediction-using-machine-learning/

9.Program using n-queen problem<br>
global N<br>
N = 4<br>
def printSolution(board):<br>
    for i in range(N):<br>
        for j in range(N):<br>
            print (board[i][j], end = " ")<br>
        print()<br>
def isSafe(board, row, col):<br>
    for i in range(col):<br>
        if board[row][i] == 1:<br>
            return False<br>
    for i, j in zip(range(row, -1, -1),<br>
            range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    for i, j in zip(range(row, N, 1),<br>
            range(col, -1, -1)):<br>
        if board[i][j] == 1:<br>
            return False<br>
    return True<br>
def solveNQUtil(board, col):<br>
    if col >= N:<br>
        return True<br>
    for i in range(N):<br>
        if isSafe(board, i, col):<br>
            board[i][col] = 1<br>
        if solveNQUtil(board, col + 1) == True:<br>
            return True<br>
        board[i][col] = 0<br>
    return False<br>
def solveNQ():<br>
    board = [ [0, 0, 0, 0],<br>
            [0, 0, 0, 0],<br>
            [0, 0, 0, 0],<br>
            [0, 0, 0, 0] ]<br>
    if solveNQUtil(board, 0) == False:<br>
        print ("Solution does not exist")<br>
        return False<br>
    printSolution(board)<br>
    return True<br>
solveNQ()<br>

Output:<br>
1 0 0 0<br> 
0 0 0 0 <br>
0 0 0 0 <br>
0 0 0 0 <br>
True<br>
***********************************************************************************************************************************************************************
10.program to find and implement A*algorthium using python:<br>
def aStarAlgo(start_node, stop_node):

open_set = set(start_node) <br>
closed_set = set()<br>
g = {} #store distance from starting node<br>
parents = {}# parents contains an adjacency map of all nodes<br>

#ditance of starting node from itself is zero<br>
g[start_node] = 0<br>
#start_node is root node i.e it has no parent nodes<br>
#so start_node is set to its own parent node<br>
parents[start_node] = start_node<br>
 
 
while len(open_set) > 0:<br>
    n = None<br>

    #node with lowest f() is found<br>
    for v in open_set:<br>
        if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):<br>
            n = v<br>
     
             
    if n == stop_node or Graph_nodes[n] == None:<br>
        pass<br>
    else:<br>
        for (m, weight) in get_neighbors(n):<br>
            #nodes 'm' not in first and last set are added to first<br>
            #n is set its parent<br>
            if m not in open_set and m not in closed_set:<br>
                open_set.add(m)<br>
                parents[m] = n<br>
                g[m] = g[n] + weight<br>

#for each node m,compare its distance from start i.e g(m) to the<br>
#from start through n node<br>
else:<br>
if g[m] > g[n] + weight:<br>
#update g(m)<br>
g[m] = g[n] + weight<br>
#change parent of m to n<br>
parents[m] = n<br>
                    #if m in closed set,remove and add to open<br>
                    if m in closed_set:<br><br>
                        closed_set.remove(m)<br>
                        open_set.add(m)<br>

    if n == None:<br>
        print('Path does not exist!')<br>
        return None<br>

    # if the current node is the stop_node<br>
    # then we begin reconstructin the path from it to the start_node<br>
    if n == stop_node:<br>
        path = []<br>

        while parents[n] != n:<br>
            path.append(n)<br>
            n = parents[n]<br>

        path.append(start_node)<br>

        path.reverse()<br>

        print('Path found: {}'.format(path))<br>
        return path<br>


    # remove n from the open_list, and add it to closed_list<br>
    # because all of his neighbors were inspected<br>
    open_set.remove(n)<br>
    closed_set.add(n)<br>

print('Path does not exist!')<br>
return None<br>
#define fuction to return neighbor and its distance<br>
#from the passed node<br>
def get_neighbors(v):<br>
if v in Graph_nodes:<br>
return Graph_nodes[v]<br>
else:
return None
#for simplicity we ll consider heuristic distances given<br>
#and this function returns heuristic distance for all nodes<br>
def heuristic(n):<br>
H_dist = {<br>
'A': 11,<br>
'B': 6,<br>
'C': 99,<br>
'D': 1,<br>
'E': 7,<br>
'G': 0,<br>
}<br>

return H_dist[n]<br>

#Describe your graph here
Graph_nodes = {<br>
'A': [('B', 2), ('E', 3)],<br>
'B': [('C', 1),('G', 9)],<br>
'C': None,<br>
'E': [('D', 6)],<br>
'D': [('G', 1)],<br>

}
aStarAlgo('A', 'G')<br>
OUTPUT:<br>
Path found: ['A', 'E', 'D', 'G']<br>
['A', 'E', 'D', 'G']<br>
**********************************************************************************************************************************************************************

11.Program to find AO*:<br>
class Graph:<br>
    def __init__(self, graph, heuristicNodeList, startNode): #instantiate graph object with graph topology, heuristic values, start node<br>
        self.graph = graph<br>
        self.H=heuristicNodeList<br>
        self.start=startNode<br>
        self.parent={}<br>
        self.status={}<br>
        self.solutionGraph={}<br>
        
    def applyAOStar(self): # starts a recursive AO* algorithm<br>
        self.aoStar(self.start, False)<br>

    def getNeighbors(self, v): # gets the Neighbors of a given node<br>
        return self.graph.get(v,'')<br>

    def getStatus(self,v): # return the status of a given node<br>
        return self.status.get(v,0)<br>

    def setStatus(self,v, val): # set the status of a given node<br>
        self.status[v]=val<br>

    def getHeuristicNodeValue(self, n):<br>
        return self.H.get(n,0) # always return the heuristic value of a given node<br>

    def setHeuristicNodeValue(self, n, value):<br>
        self.H[n]=value # set the revised heuristic value of a given node<br>

    def printSolution(self):<br>
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:",self.start)<br>
        print("------------------------------------------------------------")<br>
        print(self.solutionGraph)<br>
        print("------------------------------------------------------------")<br>

    def computeMinimumCostChildNodes(self, v): # Computes the Minimum Cost of child nodes of a given node v<br>
        minimumCost=0<br>
        costToChildNodeListDict={}<br>
        costToChildNodeListDict[minimumCost]=[]<br>
        flag=True<br>
        for nodeInfoTupleList in self.getNeighbors(v): # iterate over all the set of child node/s<br>
            cost=0<br>
            nodeList=[]<br>
            for c, weight in nodeInfoTupleList:<br>
                cost=cost+self.getHeuristicNodeValue(c)+weight<br>
                nodeList.append(c)<br>
            if flag==True: # initialize Minimum Cost with the cost of first set of child node/s<br>
                minimumCost=cost<br>
                costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child node/s<br>
                flag=False<br>
            else: # checking the Minimum Cost nodes with the current Minimum Cost<br>
                if minimumCost>cost:<br>
                    minimumCost=cost<br>
                    costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child node/s<br>
        return minimumCost, costToChildNodeListDict[minimumCost] # return Minimum Cost and Minimum Cost child node/s<br>

    def aoStar(self, v, backTracking): # AO* algorithm for a start node and backTracking status flag<br>
        print("HEURISTIC VALUES :", self.H)<br>
        print("SOLUTION GRAPH :", self.solutionGraph)<br>
        print("PROCESSING NODE :", v)<br>
        print("-----------------------------------------------------------------------------------------")<br>
        if self.getStatus(v) >= 0: # if status node v >= 0, compute Minimum Cost nodes of v<br>
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)<br>
            print(minimumCost, childNodeList)<br>
            self.setHeuristicNodeValue(v, minimumCost)<br>
            self.setStatus(v,len(childNodeList))<br>
            solved=True # check the Minimum Cost nodes of v are solved<br>
            for childNode in childNodeList:<br>
                self.parent[childNode]=v<br>
                if self.getStatus(childNode)!=-1:<br>
                    solved=solved & False<br>
            if solved==True: # if the Minimum Cost nodes of v are solved, set the current node status as solved(-1)<br>
                self.setStatus(v,-1)<br>
                self.solutionGraph[v]=childNodeList # update the solution graph with the solved nodes which may be a part of solution<br>
            if v!=self.start: # check the current node is the start node for backtracking the current node value<br>
                self.aoStar(self.parent[v], True) # backtracking the current node value with backtracking status set to true
            if backTracking==False: # check the current call is not for backtracking <br>
                for childNode in childNodeList: # for each Minimum Cost child node<br>
                    self.setStatus(childNode,0) # set the status of child node to 0(needs exploration)<br>
                    self.aoStar(childNode, False) # Minimum Cost child node is further explored with backtracking status as false<br>
                    #for simplicity we ll consider heuristic distances given<br>
print ("Graph - 1")<br>
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}<br>
graph1 = {<br>
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],<br>
    'B': [[('G', 1)], [('H', 1)]],<br>
    'C': [[('J', 1)]],<br>
    'D': [[('E', 1), ('F', 1)]],<br>
    'G': [[('I', 1)]]<br>
}

G1= Graph(graph1, h1, 'A')<br>
G1.applyAOStar()<br>
G1.printSolution()<br>
Output:<br>
Graph - 1
HEURISTIC VALUES : {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
10 ['B', 'C']
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
6 ['G']
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
10 ['B', 'C']
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : G
-----------------------------------------------------------------------------------------
8 ['I']
HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
8 ['H']
HEURISTIC VALUES : {'A': 10, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
12 ['B', 'C']
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
SOLUTION GRAPH : {}
PROCESSING NODE : I
-----------------------------------------------------------------------------------------
0 []
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': []}
PROCESSING NODE : G
-----------------------------------------------------------------------------------------
1 ['I']
HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I']}
PROCESSING NODE : B
-----------------------------------------------------------------------------------------
2 ['G']
HEURISTIC VALUES : {'A': 12, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
6 ['B', 'C']
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : C
-----------------------------------------------------------------------------------------
2 ['J']
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
6 ['B', 'C']
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G']}
PROCESSING NODE : J
-----------------------------------------------------------------------------------------
0 []
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': []}
PROCESSING NODE : C
-----------------------------------------------------------------------------------------
1 ['J']
HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 1, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
SOLUTION GRAPH : {'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J']}
PROCESSING NODE : A
-----------------------------------------------------------------------------------------
5 ['B', 'C']
FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE: A
------------------------------------------------------------
{'I': [], 'G': ['I'], 'B': ['G'], 'J': [], 'C': ['J'], 'A': ['B', 'C']}
------------------------------------------------------------
**********************************************************************************************************************************************************************

12. program to print the path from root<br>
 node to destination node for N*N-1 puzzle
 algorithm using Branch and Bound
 The solution assumes that instance of
puzzle is solvable

# Importing copy for deepcopy function
import copy

# Importing the heap functions from python
# library for Priority Queue
from heapq import heappush, heappop

# This variable can be changed to change
# the program from 8 puzzle(n=3) to 15
# puzzle(n=4) to 24 puzzle(n=5)...
n = 3

# bottom, left, top, right
row = [ 1, 0, -1, 0 ]
col = [ 0, -1, 0, 1 ]

# A class for Priority Queue
class priorityQueue:
	
	# Constructor to initialize a
	# Priority Queue
	def __init__(self):
		self.heap = []

	# Inserts a new key 'k'
	def push(self, k):
		heappush(self.heap, k)

	# Method to remove minimum element
	# from Priority Queue
	def pop(self):
		return heappop(self.heap)

	# Method to know if the Queue is empty
	def empty(self):
		if not self.heap:
			return True
		else:
			return False

# Node structure
class node:
	
	def __init__(self, parent, mat, empty_tile_pos,
				cost, level):
					
		 Stores the parent node of the
		 current node helps in tracing
		 path when the answer is found
		self.parent = parent

		 Stores the matrix
		self.mat = mat

		 Stores the position at which the
		 empty space tile exists in the matrix
		self.empty_tile_pos = empty_tile_pos

		 Storesthe number of misplaced tiles
		self.cost = cost

		 Stores the number of moves so far
		self.level = level

	This method is defined so that the
	 priority queue is formed based on
	 the cost variable of the objects
	def __lt__(self, nxt):
		return self.cost < nxt.cost

 Function to calculate the number of
 misplaced tiles ie. number of non-blank
 tiles not in their goal position
def calculateCost(mat, final) -> int:
	
	count = 0
	for i in range(n):
		for j in range(n):
			if ((mat[i][j]) and
				(mat[i][j] != final[i][j])):
				count += 1
				
	return count

def newNode(mat, empty_tile_pos, new_empty_tile_pos,
			level, parent, final) -> node:
				
	 Copy data from parent matrix to current matrix
	new_mat = copy.deepcopy(mat)

 Move tile by 1 position
	x1 = empty_tile_pos[0]
	y1 = empty_tile_pos[1]
	x2 = new_empty_tile_pos[0]
	y2 = new_empty_tile_pos[1]
	new_mat[x1][y1], new_mat[x2][y2] = new_mat[x2][y2], new_mat[x1][y1]

	 Set number of misplaced tiles
	cost = calculateCost(new_mat, final)

	new_node = node(parent, new_mat, new_empty_tile_pos,
					cost, level)
	return new_node

 Function to print the N x N matrix
def printMatrix(mat):
	
	for i in range(n):
		for j in range(n):
			print("%d " % (mat[i][j]), end = " ")
			
		print()

 Function to check if (x, y) is a valid
 matrix coordinate
def isSafe(x, y):
	
	return x >= 0 and x < n and y >= 0 and y < n

Print path from root node to destination node
def printPath(root):
	
	if root == None:
		return
	
	printPath(root.parent)
	printMatrix(root.mat)
	print()
 Function to solve N*N - 1 puzzle algorithm
using Branch and Bound. empty_tile_pos is
the blank tile position in the initial state.
def solve(initial, empty_tile_pos, final):
	
	 Create a priority queue to store live
	 nodes of search tree
	pq = priorityQueue()

	 Create the root node
	cost = calculateCost(initial, final)
	root = node(None, initial,
				empty_tile_pos, cost, 0)

	 Add root to list of live nodes
	pq.push(root)

	 Finds a live node with least cost,
	add its children to list of live
	 nodes and finally deletes it from
	 the list.
	while not pq.empty():

		 Find a live node with least estimated
		 cost and delete it form the list of
		 live nodes
		minimum = pq.pop()

		 If minimum is the answer node
		if minimum.cost == 0:
			
			 Print the path from root to
			 destination;
			printPath(minimum)
			return

		 Generate all possible children
		for i in range(n):
			new_tile_pos = [
				minimum.empty_tile_pos[0] + row[i],
				minimum.empty_tile_pos[1] + col[i], ]
				
			if isSafe(new_tile_pos[0], new_tile_pos[1]):
				
				 Create a child node
				child = newNode(minimum.mat,
								minimum.empty_tile_pos,
								new_tile_pos,
								minimum.level + 1,
								minimum, final,)

				 Add child to list of live nodes
				pq.push(child)

Driver Code

 Initial configuration
 Value 0 is used for empty space
initial = [ [ 1, 2, 3 ],
			[ 5, 6, 0 ],
			[ 7, 8, 4 ] ]

 Solvable Final configuration
Value 0 is used for empty space
final = [ [ 1, 2, 3 ],
		[ 5, 8, 6 ],
		[ 0, 7, 4 ] ]

 Blank tile coordinates in
 initial configuration
empty_tile_pos = [ 1, 2 ]

 Function call to solve the puzzle
solve(initial, empty_tile_pos, final)

This code is contributed by Kevin Joshi
