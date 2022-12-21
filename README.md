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

#5.Write a Program to Implement Best First Search using Python.<br>
from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>

 # for implementing BFS | returns path having lowest cost<br>
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

  # for adding edges to graph<br><br>
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
# Tic-Tac-Toe Program using<br>
# random number in Python<br>
 
# importing all necessary libraries<br>
import numpy as np<br>
import random<br>
from time import sleep<br>
 
# Creates an empty board<br>
 
 
def create_board():<br>
    return(np.array([[0, 0, 0],<br>
                     [0, 0, 0],<br>
                     [0, 0, 0]]))<br>
 
# Check for empty places on board<br>
 
 
def possibilities(board):<br>
    l = []<br>
 
    for i in range(len(board)):<br>
        for j in range(len(board)):<br>
 
            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>
 
# Select a random place for the player<br>
 
 <br>
def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player<br>
    return(board)<br>
 
# Checks whether the player has three<br>
# of their marks in a horizontal row<br>
 
 
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
 
# Checks whether the player has three<br>
# of their marks in a vertical row<br>
 
 
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
 
# Checks whether the player has three<br>
# of their marks in a diagonal row<br>
 
 
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
 
# Evaluates whether there is<br>
# a winner or a tie<br>
 
 
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
 
# Main function to start the game<br>
 
 
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
 
 
# Driver Code<br>
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

# Python3 program to print the path from root
# node to destination node for N*N-1 puzzle
# algorithm using Branch and Bound
# The solution assumes that instance of
# puzzle is solvable

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

# Function to calculate the number of
# misplaced tiles ie. number of non-blank
# tiles not in their goal position
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

# Function to print the N x N matrix
def printMatrix(mat):
	
	for i in range(n):
		for j in range(n):
			print("%d " % (mat[i][j]), end = " ")
			
		print()

# Function to check if (x, y) is a valid
# matrix coordinate
def isSafe(x, y):
	
	return x >= 0 and x < n and y >= 0 and y < n

# Print path from root node to destination node<br>
def printPath(root):<br>
	
	if root == None:
		return
	
	printPath(root.parent)<br>
	printMatrix(root.mat)<br>
	print()<br>

# Function to solve N*N - 1 puzzle algorithm<br>
# using Branch and Bound. empty_tile_pos is<br>
# the blank tile position in the initial state.<br>
def solve(initial, empty_tile_pos, final):<br>
	
	# Create a priority queue to store live
	# nodes of search tree
	pq = priorityQueue()

	# Create the root node
	cost = calculateCost(initial, final)
	root = node(None, initial,
				empty_tile_pos, cost, 0)

	# Add root to list of live nodes
	pq.push(root)

	# Finds a live node with least cost,
	# add its children to list of live
	# nodes and finally deletes it from
	# the list.
	while not pq.empty():

		# Find a live node with least estimated
		# cost and delete it form the list of
		# live nodes
		minimum = pq.pop()

		# If minimum is the answer node
		if minimum.cost == 0:
			
			# Print the path from root to
			# destination;
			printPath(minimum)
			return

		# Generate all possible children
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

# Driver Code

# Initial configuration
# Value 0 is used for empty space
initial = [ [ 1, 2, 3 ],
			[ 5, 6, 0 ],
			[ 7, 8, 4 ] ]

# Solvable Final configuration
# Value 0 is used for empty space
final = [ [ 1, 2, 3 ],
		[ 5, 8, 6 ],
		[ 0, 7, 4 ] ]

# Blank tile coordinates in
# initial configuration
empty_tile_pos = [ 1, 2 ]

# Function call to solve the puzzle
solve(initial, empty_tile_pos, final)<br>

