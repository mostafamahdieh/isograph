import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


public class CheckBetweennessOutput {
	public static void main(String[] args) throws FileNotFoundException {
		String fileName = args[0];
		Scanner scanner = new Scanner(new File(fileName));
		scanner.nextLine();
		
		int pos = 0;
		while (scanner.hasNextLine()) {
			pos++;
			System.out.println("line: " + pos);
			String[] s = scanner.nextLine().split(" ");
			String[] s2 = scanner.nextLine().split(" ");
			assert(s.length == 2);
			assert(s2.length == 1);
		}
	}
}
