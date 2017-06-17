package bohnenspiel;

import java.awt.image.ByteLookupTable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import bohnenspiel.BohnenspielBoardState;
import bohnenspiel.BohnenspielMove;
import bohnenspiel.BohnenspielMove.MoveType;
import bohnenspiel.BohnenspielPlayer;
import student_player.mytools.MyTools;

/** A Hus player submitted by a student. */
public class GreedyBohnenspielPlayer extends BohnenspielPlayer {

	static final int MINIMAX_TIME = 450;
	static final int RETURN_TIME = 600;
	static final int MAX_DEPTH = 10;
	static final String STUDENT_ID = "260563512";
	static final int WIN_THRESHOLD = 36;


    /** You must modify this constructor to return your student number.
     * This is important, because this is what the code that runs the
     * competition uses to associate you with your agent.
     * The constructor should do nothing else. */
    //public StudentPlayer() { super(STUDENT_ID); }

    /** This is the primary method that you need to implement.
     * The ``board_state`` object contains the current state of the game,
     * which your agent can use to make decisions. See the class
bohnenspiel.RandomPlayero
     * for another example agent. */
    public BohnenspielMove chooseMove(BohnenspielBoardState board_state) {
    	
        // Get the legal moves for the current board state.
        ArrayList<BohnenspielMove> moves = board_state.getLegalMoves();
    	
        double best_score = MinMaxType.MAX.defaultValue;
        BohnenspielMove best_move = null;
        
        for(BohnenspielMove move: moves) {
        	
        	BohnenspielBoardState clone = (BohnenspielBoardState) board_state.clone();
        	clone.move(move);
        	
        	double candidate_score = alphabeta(0, clone, MinMaxType.MAX.defaultValue, MinMaxType.MIN.defaultValue, MinMaxType.MIN);
        	
        	if (candidate_score > best_score) {
        		best_score = candidate_score;
        		best_move = move;
        	}
        	
        	System.out.println(candidate_score);
        }

        return best_move;
    }
    
    private double alphabeta(int recursionDepth, BohnenspielBoardState board_state, double a, double b,  MinMaxType minMaxType) {
    	
    	if (recursionDepth == MAX_DEPTH)
    		return heuristicScore(board_state.getScore(player_id), board_state.getScore(opponent_id));
    	
    	double best_score = minMaxType.defaultValue;
    	
        ArrayList<BohnenspielMove> moves = board_state.getLegalMoves();
        
        for (BohnenspielMove move: moves) {
        	
        	BohnenspielBoardState clone = (BohnenspielBoardState) board_state.clone();
        	clone.move(move);
        	
        	double candidate_score = alphabeta(recursionDepth + 1, clone, a, b, minMaxType.opposite());
        	
        	if (minMaxType.equals(MinMaxType.MAX)) {
        		best_score = Math.max(best_score, candidate_score);
        		a = Math.max(a, best_score);
        		if (a >= b)
        			return a;
        		
        		return best_score;
        		
        	} else {
        		best_score = Math.min(best_score, candidate_score);
        		b = Math.min(b, best_score);
        		if (b <= a) {
        			return b;
        		}
        	}
        }
        
        return best_score;

    }
    
    private double heuristicScore(int playerScore, int opponentScore) {
    	return (playerScore - opponentScore) / 72.0;
    }
   
    private enum MinMaxType {
    	
    	MIN(2.0),
    	MAX(-2.0);
    	
    	private final double defaultValue;

    	
    	private MinMaxType(double defaultValue) {
    		this.defaultValue = defaultValue;
    	}
    	
    	private MinMaxType opposite() {
    		switch (this) {
	    		case MIN: return MAX;
	    		case MAX: return MIN;
	    		default: return null;
    		}
    	}
    	
    }
    
//  private ArrayList<BohnenspielMove> prune(BohnenspielBoardState board_state, ArrayList<BohnenspielMove> moves, MinMaxType minMaxType) {
//	
//	ArrayList<MoveScore> moveScores = new ArrayList<>();
//	ArrayList<BohnenspielMove> pruned_moves = new ArrayList<>();
//	
//	double scoreFirst = minMaxType.defaultValue;
//	double scoreSecond = minMaxType.defaultValue;
//	
//	// takes only moves that score top PRUNING_FACTOR
//	for (BohnenspielMove move: moves) {
//		
//		BohnenspielBoardState clone = (BohnenspielBoardState) board_state.clone();
//		clone.move(move);
//		double score = heuristicScore(clone.getScore(player_id), clone.getScore(opponent_id));
//		
//		if (minMaxType.comp(score, scoreFirst)) {
//			scoreFirst = score;
//		} else if (minMaxType.comp(score, scoreSecond)) {
//			scoreSecond = score;
//		}
//		
//		moveScores.add(new MoveScore(score, move));
//	}
//	
//	for (MoveScore moveScore: moveScores) {
//		if (minMaxType.comp(moveScore.score, scoreFirst)) {
//			pruned_moves.add(moveScore.move);
//		}
//	}
//	
//	if(pruned_moves.size() < PRUNING_FACTOR) {
//    	for (MoveScore moveScore: moveScores) {
//    		if (minMaxType.comp(moveScore.score, scoreSecond)) {
//    			pruned_moves.add(moveScore.move);
//    		}
//    	}
//	}
//	
//	// reduces size down to PRUNING_FACTOR 
//	if(pruned_moves.size() > PRUNING_FACTOR) {
//		
//		ArrayList<BohnenspielMove> new_pruned = new ArrayList<>();
//		Random rand = new Random();
//		
//		for(int i = 0; i < PRUNING_FACTOR; i++) {
//			new_pruned.add(pruned_moves.remove(rand.nextInt(pruned_moves.size())));
//		}
//		
//		pruned_moves = new_pruned;
//	}
//	
//	return pruned_moves;
//}
    
    
//    private class MoveScore {
//    	double score;
//    	BohnenspielMove move;
//    	
//    	public MoveScore(double score, BohnenspielMove move) {
//    		this.score = score;
//    		this.move = move;
//		}
//    }


//private double monteCarlo(BohnenspielBoardState board_state, MinMaxType minMaxType) {
//	
//	// System.out.println("depth " + monteDepth++ + " score " + board_state.getScore(player_id) + " " + board_state.getScore(opponent_id));
//    
//	// Get the legal moves for the current board state.
//    ArrayList<BohnenspielMove> moves = board_state.getLegalMoves();
//
//    double score = 0;
//    double bestscore = minMaxType.defaultValue;
//    BohnenspielMove bestMove = null;
//	
//    for (BohnenspielMove move : moves) {
//
//        BohnenspielBoardState clone = (BohnenspielBoardState) board_state.clone();
//        clone.move(move);
//        score = clone.getScore(minMaxType.player(player_id, opponent_id));
//        
//        if(score > WIN_THRESHOLD)
//        	return heuristicScore(clone.getScore(player_id), clone.getScore(opponent_id));
//        
//        if (minMaxType.comp(score, bestscore)) {
//	        bestMove = move;
//	        bestscore = score;
//        }
//    }
//    
//    if (bestMove == null) {
//    	bestMove = moves.get(new Random().nextInt(moves.size()));
//    }
//
//    board_state.move(bestMove);
//	
//	return monteCarlo(board_state, minMaxType.opposite());
//
//}

    
}