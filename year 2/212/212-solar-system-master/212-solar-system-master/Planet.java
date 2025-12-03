public class Planet{
    
    public void Planet(SolarSystem SolarSystem, double distance, double angle, double diameter, String col, double rotation) {
        this.distance = distance;
        this.angle = angle;
        this.diameter = diameter;
        this.col = col;
        this.rotation = rotation;
    }

    double distance;  
    double angle; 
    double diameter; 
    String col; 
    double rotation;

    double getDistance(){
        return this.distance;
    }

    double getAngle(){
        return this.angle;
    }

    double getDiameter(){
        return this.diameter;
    }

    String getCol(){
        return this.col;
    }


    void setDistance(double distance){
        this.distance = distance;
    }

    void setAngle(double angle){
        this.angle = angle;
        if (this.angle == 359){
            this.angle = 0;
        }
    }

    void setDiameter(double diameter){
        this.diameter = diameter;
    }

    void setCol(String col){
        this.col = col;
    }

    double getRotation(){
        return this.rotation;
    }

    void setRotation(){
        this.rotation = this.rotation + this.rotation;
    }

    void move(){
        this.setAngle(this.getRotation());
    }
}
