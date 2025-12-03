import java.rmi.Remote;
import java.rmi.RemoteException;
import java.security.PublicKey;

import javax.crypto.SealedObject;

public interface Auction extends Remote {
    
    int register(String email, PublicKey pkey) throws RemoteException;

    ChallengeInfo challenge(int userID, String clientChallenge) throws RemoteException;

    TokenInfo authenticate(int userID, byte[] signature) throws RemoteException;

    AuctionItem getSpec(int userID, int itemID, String token) throws RemoteException;

    int newAuction(int userID, AuctionSaleItem item, String token) throws RemoteException;

    AuctionItem[] listItems(int userID, String token) throws RemoteException;

    AuctionResult closeAuction(int userID, int itemID, String token) throws RemoteException;
    
    boolean bid(int userID, int itemID, int price, String token) throws RemoteException;
}