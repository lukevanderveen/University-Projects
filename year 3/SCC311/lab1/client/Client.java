import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;

public class Client{
     public static void main(String[] args) {
       if (args.length < 1) {
       System.out.println("Usage: java Client n");
       return;
       }

         int n = Integer.parseInt(args[0]);
         try {
               String name = "myserver";
               Registry registry = LocateRegistry.getRegistry("localhost");
               ICalc server = (ICalc) registry.lookup(name);
  
                int result = server.factorial(n);
                System.out.println("result is " + result);
              }
              catch (Exception e) {
               System.err.println("Exception:");
               e.printStackTrace();
               }
      }
}