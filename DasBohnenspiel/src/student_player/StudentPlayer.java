package student_player;
import java.util.ArrayList;
import bohnenspiel.BohnenspielBoardState;
import bohnenspiel.BohnenspielMove;
import bohnenspiel.BohnenspielPlayer;
import student_player.mytools.MyTools;


/** A Hus player submitted by a student. */
public class StudentPlayer extends BohnenspielPlayer {

	/** You must modify this constructor to return your student number.
	 * This is important, because this is what the code that runs the
	 * competition uses to associate you with your agent.
	 * The constructor should do nothing else. */


	public StudentPlayer() { super("260581958"); }

	/** This is the primary method that you need to implement.
	 * The ``board_state`` object contains the current state of the game,
	 * which your agent can use to make decisions. See the class
	 * bohnenspiel.RandomPlayer
	 * for another example agent. */


	public BohnenspielMove chooseMove(BohnenspielBoardState board_state){
		//long startTime = System.nanoTime();
		
		//create root node
		ArrayList<MyTools.Node> nextMoves = new ArrayList<MyTools.Node>();
		int score = board_state.getScore(1);
		MyTools.Node root = new MyTools.Node(score,null, nextMoves);
		
		//int depthIncrease = (int)Math.floor(board_state.getTurnNumber()/5.0);
		int depth = 5;// + depthIncrease;
		//System.out.println("Depth: "+depth);
		
		//Initialize hyper-parameters, for each of the heuristics
		
		//weighting captures k2 < k4 < k6 with strict inequalities
		int k2 = 3;
		int k4 = 2*k2+1; 
		int k6 = 3*k2+3; 
		
		//to weight moves leaving opponent with all empty pits
		int k_Max = 25; 
		
		//threshold of what amount of beans to favor
		int k_Hoard = 6;
		
		//to weight moves which result in getting more than 36 points
		int k_win = 10000;
	
		
		//make the graph starting from the root, and determine best move
		MyTools.makeGraph(root,depth,board_state,k2, k4, k6, k_Max, k_Hoard, k_win);
		int maxValue = MyTools.minimax2(root, depth, true);
		
		//Initialize first move
		ArrayList<BohnenspielMove> moves = board_state.getLegalMoves();
		BohnenspielMove move = moves.get(0);
		
		//update with the best value move found
		for(int i =0; i<nextMoves.size(); i ++){
			if(root.getMoves().get(i).getValue() == maxValue){
				move = root.getMoves().get(i).getMove();
			}
		}
		//long endTime = System.nanoTime();
		
		//Time to make a move should be less than 0.7 seconds
		//System.out.println("Time to find move: "+(endTime-startTime)/1000000000.0);
		return move;	
	}
}