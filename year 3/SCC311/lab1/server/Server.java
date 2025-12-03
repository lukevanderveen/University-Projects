import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class Server implements ICalc{
    public Server() throws RemoteException {
        super();
     }
  
    public int factorial(int n) {
     System.out.println("client request handled");
     if (n == 0 || n == 1) {
         return 1;
     }
     return n * factorial(n - 1);
     }
  
     public static void main(String[] args) {
        try {
         Server s = new Server();
         System.out.println("here 1");
         String name = "myserver";
         System.out.println("here 2");
         ICalc stub = (ICalc) UnicastRemoteObject.exportObject(s, 0);
         System.out.println("here 3");
         Registry registry = LocateRegistry.getRegistry();
         System.out.println("here 4");
         registry.rebind(name, stub);
         System.out.println("here 5");
         System.out.println("Server ready");
         System.out.println("here 6");
        } catch (RemoteException e) {
            System.out.println("RemoteException: ");
            e.printStackTrace();
        }catch (Exception e) {
         System.err.println("Exception:");
         e.printStackTrace();
        }
    }
}
