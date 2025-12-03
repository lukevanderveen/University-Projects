public class InventoryThread extends Thread {

    private int mode;
    private Inventory i;

    public InventoryThread(int mode, Inventory i){
        this.mode = mode;
        this.i = i;
        
    }

    public int getMode(){
        return this.mode;
    }

    //methods for adding and removing 
    public void add(Inventory i){
        i.increment();
        System.out.println("Added. Inventory size = "+ i.getTotal());
    }

    public void remove(Inventory i){
        i.decrement();
        System.out.println("Removed. Inventory size = "+ i.getTotal());
    }

    
    @Override
    public void run(){
        if (this.getMode() == 1){
            add(i);
        }else if (this.getMode() == 0){
            remove(i);
        }
    }
}