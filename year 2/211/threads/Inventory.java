public class Inventory {

    //inventroy class to keep track of the inventroy's quantity
    private int total = 0;

    public Inventory(){
    }

    public int getTotal(){
        return total;
    }
    
    public synchronized void increment(){
        total++;
    }

    public synchronized void decrement(){
        total--;
    }
}
