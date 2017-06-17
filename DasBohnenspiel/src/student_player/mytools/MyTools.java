package student_player.mytools;
import java.util.ArrayList;
import bohnenspiel.BohnenspielBoardState;
import bohnenspiel.BohnenspielMove;
public class MyTools {

	public static double getSomething(){
		return Math.random();
	}
	public static class Node {
		int value;
		BohnenspielMove move;
		ArrayList<Node> nextMoves;

		public Node(int value, BohnenspielMove move, ArrayList<Node> nextMoves){
			this.value = value;
			this.move = move;
			this.nextMoves = nextMoves;
		}
		public ArrayList<Node> getMoves(){
			return this.nextMoves;
		}
		public int getValue(){
			return this.value;
		}
		public BohnenspielMove getMove(){
			return this.move;
		}
	}

	//print each value of the graph created DFS style
	public static void testTraversal(Node root){
		System.out.println(root.value);
		if(root.nextMoves != null){
			for(int i =0; i<root.nextMoves.size(); i++){
				testTraversal(root.nextMoves.get(i));
			}
		}
		return;
	}
	//create the first root node then create the new graph
	//does not return anything, merely creates a graph with depth given starting for root provided

	public static void makeGraph(Node root, int depth, BohnenspielBoardState board_state, int k2, int k4, int k6, int k_Max, double k_Hoard_Power, int k_Won){
		int player_id = board_state.getTurnPlayer();

		//for each legal move create a new node
		ArrayList<BohnenspielMove> moves = board_state.getLegalMoves();

		//if depth null or no available moves
		if(depth == 0 || moves.size()==0){
			return;
		}

		int score;
		BohnenspielBoardState new_board_state;

		for(int i =0; i<moves.size(); i ++){

			BohnenspielBoardState cloned_board_state = (BohnenspielBoardState) board_state.clone();
			BohnenspielMove move1 = moves.get(i);

			cloned_board_state.move(move1);
			score=cloned_board_state.getScore(player_id);
			new_board_state = cloned_board_state; //after move i


			//make new root to subgraph and do a recursive call
			ArrayList<Node> nextMoves = new ArrayList<Node>();
			Node subRoot = new Node(score, moves.get(i), nextMoves);

			//add value and move to sub root, player is either 1 or -1
			subRoot.value = score + captureValue(board_state, move1, player_id, k2, k4, k6, k_Max, k_Hoard_Power, k_Won);
			subRoot.move = moves.get(i);

			//add new node to root
			root.nextMoves.add(subRoot); 

			//continue graph starting at subgraph
			makeGraph(subRoot, depth-1, new_board_state, k2, k4, k6, k_Max, k_Hoard_Power, k_Won);	
		}    	
		return;
	}
	public static int captureValue(BohnenspielBoardState board_state, BohnenspielMove move, int player, int k2, int k4, int k6, int k_Max, double k_Hoard, int k_Won){

		//get current pits
		int[][] pits = board_state.getPits();
		int[] p_pits = pits[player];
		//int[] op_pits = pits[(player+1)%2];

		//attempt move
		BohnenspielBoardState cloned_board_state = (BohnenspielBoardState) board_state.clone();
		cloned_board_state.move(move);
		int score=cloned_board_state.getScore(player);

		//If a score pushes a player over the 36 point mark they've already won.
		int kWin = 0;
		if(score >= 36){
			kWin = k_Won;
		}

		//get pits after move
		int[][] pitsC = cloned_board_state.getPits();
		int[] p_pitsC = pitsC[player];
		int[] op_pitsC = pitsC[(player+1)%2];

		//capture benefit store in k
		int k=0;
		
		int kHoard = 0;
		int dif;
		for(int i =0; i < p_pitsC.length; i++){
			if(p_pitsC[i] >= k_Hoard){
				k += p_pitsC[i]-k_Hoard;
			}
			
			dif = p_pitsC[i] - p_pits[i];
			//if we have captured on that pit
			if(p_pitsC[i] == 0){

				//pit was captured with a score of +2
				if(dif == 1 && p_pits[i] == 1){
					k+=k2;
				}
				//pit was captured with a score of +4
				//I increase the weight of capturing 4 beans as to 
				//differentiate it from moving a single bean into another single and  capturing
				else if(dif == 3 && p_pits[i] == 3){
					k+=k4;
				}
				//pit was captured with a score of +6
				else if(dif == 5 && p_pits[i] == 3){
					k+=k6;
				}
			}

			//pit was captured with a score of +2
			if(op_pitsC[i] == 1){
				k+=-k2;
			}
			//pit was captured with a score of +4
			//I increase the weight of capturing 4 beans as to 
			//differentiate it from moving a single bean into another single and  capturing
			else if(op_pitsC[i] == 3){
				k+=-k4;
			}
			//pit was captured with a score of +6
			else if(op_pitsC[i] == 5){
				k+=-k6;
			}
		}

		//final move benefit, if opponent
		int kMax = k_Max;
		for(int i =0; i< op_pitsC.length; i++){
			//if the opponent only has empty pits they 
			//must skip or the player can capture all beans on their side. 
			if(op_pitsC[i] != 0){
				kMax = 0;
			}
		}
		return k+kMax+kHoard+kWin;
	}

	//Performs a minimax search to a given depth
	//heuristic will be stored in value of the node
	//nodes have move and value
	public static int minimax(Node node, int depth, boolean maximizingPlayer, BohnenspielBoardState board_state){

		//Initialize variables to be used later
		//int player_id = board_state.getTurnPlayer();

		//Score to hold the points for a particular move, value to store the true value of the move
		//if the value is on a leaf then both score and value are the same for that move
		//int score;
		int value;

		//max value holds best moves
		int maxValue;

		//if no more moves are available or desired depth is reached
		if(depth == 0 || node.nextMoves.size()==0){
			return node.value;
		}

		//Maximizing for the student player
		if (maximizingPlayer){
			maxValue = -100000;

			//iterate through each of the moves, nextMoves is a collection of nodes
			for(int i = 0; i < node.nextMoves.size(); i++){

				//Clone the current board
				BohnenspielBoardState cloned_board_state = (BohnenspielBoardState) board_state.clone();

				//Retrieve one of the legal moves, store in the graph
				BohnenspielMove move1 = node.nextMoves.get(i).move;

				//perform the move from above and extract the new score
				cloned_board_state.move(move1);
				//score=cloned_board_state.getScore(player_id);

				//update the node's value using nodes further down the graph
				value = minimax(node.nextMoves.get(i), depth-1, false, cloned_board_state);

				//update the nodes value
				node.value = value;

				//track the best move from each node.
				if (value>maxValue)
				{
					maxValue=value;
				}

				//maxValue = max(maxValue, value);
			}
			return maxValue;

			//Minimizing the maximum move for the opponent
		}else{
			maxValue = 100000;
			for(int j = 0; j < node.nextMoves.size(); j++){

				//clone the current board
				BohnenspielBoardState cloned_board_state = (BohnenspielBoardState) board_state.clone();

				//Retrieve one of the legal moves, store in graph
				BohnenspielMove move1 = node.nextMoves.get(j).move;

				//Perform the move from the above and extract the new score
				cloned_board_state.move(move1);
				//score=cloned_board_state.getScore(player_id);

				//update the node's value using nodes further down the graph
				value = minimax(node.nextMoves.get(j), depth-1, true, cloned_board_state);

				//track the move with the worst value
				if (value<maxValue)
				{
					maxValue=value;
				}
				//maxValue = min(maxValue, value);    				
			}
			return maxValue;
		}
	}
	//Performs a minimax search to a given depth
	//heuristic will be stored in value of the node
	//nodes have move and value
	public static int minimax2(Node node, int depth, boolean maximizingPlayer){

		//initialize some variable to hold values
		int value;
		int maxValue;

		//if no more moves are available or desired depth is reached
		if(depth == 0 || node.nextMoves.size()==0){
			return node.value;
		}

		//Maximizing for the student player
		if (maximizingPlayer){
			maxValue = -100000;

			//iterate through each of the moves, nextMoves is a collection of nodes
			for(int i = 0; i < node.nextMoves.size(); i++){
				//update the node's value using nodes further down the graph
				value = minimax2(node.nextMoves.get(i), depth-1, false);

				//track the best move from each node.
				if (value>maxValue)
				{
					maxValue=value;
				}

				//maxValue = max(maxValue, value);
			}
			node.value = maxValue;
			return maxValue;

			//Minimizing the maximum move for the opponent
		}else{
			maxValue = 100000;
			for(int j = 0; j < node.nextMoves.size(); j++){
				//update the node's value using nodes further down the graph
				value = minimax2(node.nextMoves.get(j), depth-1, true);

				//track the move with the worst value
				if (value<maxValue)
				{
					maxValue=value;
				}
				//maxValue = min(maxValue, value);    				
			}
			node.value = maxValue;
			return maxValue;
		}
	}
}