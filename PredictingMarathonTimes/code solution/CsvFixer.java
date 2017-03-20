package csvfix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;


public class CsvFixer 
{
	public static void main(String[] args)
	{
		String temp = "";
		String csvFile = "/Users/Gavin/workspace/csvfix/src/csvfix/a.csv";
        BufferedReader br = null;
        String line = "";
        String csvSplit = ":";
        PrintWriter w = null;
        int count = 0;
        
        try {

            br = new BufferedReader(new FileReader("C:\\Users\\Gavin\\Documents\\aadata.csv"));
            w = new PrintWriter("output.csv");
           
            while ((line = br.readLine()) != null) {
            	count++;
            	System.out.println(line);
                String[] time = line.split(csvSplit);

                int sum = Integer.parseInt(time[0])*60 + Integer.parseInt(time[1]);
                String temp1 = Integer.toString(sum);
                if(temp1.equals(""))
                {
                	break;
                }
                System.out.println(temp1);
                w.print(temp1+"\n");
            }
            w.close();
            System.out.println(count);
        } catch (IOException e) 
        {
            e.printStackTrace();
        } finally 
        {
            if (br != null) 
            {
                try 
                {
                    br.close();
                } catch (IOException e) 
                {
                }
            }
        }
        

	}
}
