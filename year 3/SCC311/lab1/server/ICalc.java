import java.rmi.Remote;
import java.rmi.RemoteException;

public interface ICalc extends Remote{
  int factorial(int n) throws RemoteException;
}
