
public class Neural {
	static double n=0.15;	// learning rate
	static int N=6; //number of nodes (including input nodes)
	static int M=8;	// number of connections
	static int K=2; // number of input variables
	static int P=2; // number of data series
	static int L=1; // number of outputs
	static int EPOCHS=1; // number of epochs
	
	
	//declare and create array of nodes
	private Node[] nodes=new Node[N];
	//declare and create array of connections
	private Connection[] connections=new Connection[M];
	
	private static double inputs[][]=new double[K][P];
	private double desirableOutputs[][]=new double[P][L];
	
	// Empty Constructor
	public Neural(){
		inputs[0][0]=1;
		inputs[0][1]=0;
		inputs[1][0]=1;
		inputs[1][1]=1;
		
		// declare desirable output
		desirableOutputs[0][0]=1;
		desirableOutputs[1][0]=0;
		// creation of the nodes
		nodes[0]=new Node(-1,0); 	//input node
		nodes[1]=new Node(-2,0);	// input node
		nodes[2]=new Node(1,0,0,1);	// hidden layer node
		nodes[3]=new Node(2,0,0,1);	// hidden layer node
		nodes[4]=new Node(3,0,0,1);	// hidden layer node
		nodes[5]=new Node(4,0,1,2);	// output node
		// creation of the connections
		connections[0]=new Connection(nodes[0],nodes[2],0.14);
		connections[1]=new Connection(nodes[0],nodes[3],-0.2);
		connections[2]=new Connection(nodes[1],nodes[3],0.5);
		connections[3]=new Connection(nodes[1],nodes[5],-0.3);
		connections[4]=new Connection(nodes[1],nodes[4],0.15);
		connections[5]=new Connection(nodes[2],nodes[5],0.5);
		connections[6]=new Connection(nodes[3],nodes[5],-0.33);
		connections[7]=new Connection(nodes[4],nodes[5],0.4);
	}
	
	// forward passing for k data series
	public void forward(int s){
		int i,j;
		double sum;
		// For the input nodes the output is equal to the input
		// We have K input nodes
		for(i=0;i<K;i++){
			nodes[i].setOutput(inputs[s][i]);
		}
		System.out.println("Forward:");
		// For all the other nodes
		for(i=K;i<N;i++){
			sum=0;
			// check the connections
			for(j=0;j<M;j++)
				if(connections[j].getEndId()==nodes[i].getId()) {
					
					sum=sum+connections[j].getWeight()*connections[j].getStartOutput();
				}
			sum=sum-nodes[i].getThreshold();	
			nodes[i].setSum(sum);
			nodes[i].fire();
			System.out.println("output" +nodes[i].getId()+":"+nodes[i].getOutput());
		}
	}
	
	// Calculates output error (beta-out)
	public void calcError(int s){
		int i;
		double e;
		System.out.println("Error (beta-out):");
		// calculate error for the output nodes
		for(i=N-L;i<N;i++){
			e=desirableOutputs[s][N-1-i]-nodes[i].getOutput();
			System.out.println(e);
			nodes[i].setError(e);
		}
	}
	
	public void backward(){
		int i,j;
		
		System.out.println("Betas:");
		for(i=N-1;i>=K;i--){
			calcBeta(nodes[i]);
			System.out.println("node "+nodes[i].getId()+":"+nodes[i].getBeta());
			//nodes[i].updateThreshold(n);
		}
		for(j=0;j<M;j++){
			connections[j].updateWeight(n);
			System.out.println("weight("+connections[j].getStartId()+","+connections[j].getEndId()+")="+connections[j].getWeight());
		}
		
	}
	
	// Calculate betas
	private void calcBeta(Node n){
		// if it is an output node
		double y,e,sum=0;
		int j;
		y=n.getOutput();
		if(n.getType()==2){
			e=n.getError();
			// sigmoid
			if(n.getFunction()==0)
				n.setBeta(e*y*(1-y));
			// linear
			else
				n.setBeta(e);
		}
		// hidden layer
		else if(n.getType()==1){
			// check the out-connections
			for(j=0;j<M;j++){
				if(connections[j].getStartId()==n.getId()) {
					sum=sum+connections[j].getWeight()*connections[j].getEndNode().getBeta();
				}
			}
			// sigmoid
			if(n.getFunction()==0){
				n.setBeta(y*(1-y)*sum);
			}
			// linear
			else
				n.setBeta(sum);
		}
			
	}
		
	public static void main(String[] args){
		int s,ep;
		Neural net1=new Neural();
		for(ep=1;ep<=EPOCHS;ep++){
			System.out.println("EPOCH-"+ep);
			for(s=0;s<P;s++){
				System.out.println("Data Series:"+s);
				net1.forward(s);
				net1.calcError(s);
				net1.backward();
			}
		}
			
	}
	
	
}

class Node {
	// the node's id
	private int id;
	// This is the threshold 
	private double threshold;
	// function=0 means sigmoid
	// function=1 means linear function
	private int function;
	// output of the node
	private double output;
	// sum calculated for the node
	private double sum;
	// type=0 means the node is an input node
	// type=1 means it is node of the hidden layer
	// type=2 means it is an output node
	private int type;
	// error for the output nodes
	double error;
	private double beta;
	// Constructor
	public Node(int nodeId){
		id=nodeId;
		threshold=0;
		function=0;
		type=1;
	}
	
	// 2nd constructor that takes as input the
	// node's threshold, funtion (sigmoid or linear)
	// and type of the node (0 or 1 or 2)
	public Node(int nodeId, double thres,int f, int t){
		id=nodeId;
		threshold=thres;
		function=f;
		type=t;
	}
	
	//3rd constructor
	public Node( int nodeId,int t){
		id=nodeId;
		threshold=0;
		function=0;
		type=t;
	}
	// get the ID of the node
	public int getId(){
		return id;
	}
	
	// get the type of the node
	public int getType(){
		return type;
	}
	
	// set the output
	public void setOutput(double o){
		output=o;
	}
	

	// set the node's sum
	public void setSum(double s){
		sum=s;
	}
	// get the node's sum
	public double getSum(){
		return sum;
	}
	
	
	// returns the output
	public double getOutput(){
		return output;
	}
	
	// returns the node's threshold
	public double getThreshold(){
		return threshold;
	}
	// update node's threshold
	public void updateThreshold(double n){
		threshold=threshold+n*(-1)*this.getBeta();
	}
	
	public void fire(){
		if(function == 0)
			output=sigmoid();
		else
			output=linear();
	}
	
	private double sigmoid(){
		return 1/(1+java.lang.Math.exp(-sum));
	}
	
	private double linear(){
		return sum;
	}
	
	// returns the function type
	public int getFunction(){
		return function;
	}
	// set the delta for the node
	public void setBeta(double d){
		beta=d;
	}
	// get the delta for the node
	public double getBeta(){
		return beta;
	}
	// sets the error for the node
	public void setError(double e){
		error=e;
	}
	// returns the error of the node
	public double getError(){
		return error;
	}

}


class Connection{
	private double weight;
	// start and end of the connection
	private Node start,end;
	
	//constructor
	public Connection(Node n1,Node n2,double w){
		start=n1;
		end=n2;
		weight=w;
	}
	// returns the ID of the End Node
	public int getEndId(){
		return end.getId();
	}
	// returns the End Node
	public Node getEndNode(){
		return end;
	}
	// returns the ID of the Start Node
	public int getStartId(){
		return start.getId();
	}
	// returns the weight of the connection
	public double getWeight(){
		return weight;
	}
	// return the start node
	public  double getStartOutput(){
		return start.getOutput();
	}
	public void updateWeight(double n){
		weight=weight+n*start.getOutput()*end.getBeta();
	}
}
	

