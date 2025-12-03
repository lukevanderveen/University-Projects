public class Moon{

    double diameter; 
    String col; 
    double centreOfRotationDistance; 
    double centreOfRotationAngle;

    public void Moon(Planet planet, double diameter, String col, double centreOfRotationDistance, double centreOfRotationAngle){
        this.diameter = diameter;
        this.col = col;
        this.centreOfRotationDistance = centreOfRotationDistance;
        this.centreOfRotationAngle = centreOfRotationAngle;
    }

    void setCentreOfRotationDistance(double centreOfRotationDistance){
        this.centreOfRotationDistance = centreOfRotationDistance;
    }

    void setCentreOfRotationAngle(double centreOfRotationAngle){
        this.centreOfRotationAngle = centreOfRotationAngle;
        if (this.centreOfRotationAngle == 359){
            this.centreOfRotationAngle = 0;
        }
    }

    double getCentreOfRotationDistance(){
        return this.centreOfRotationDistance;
    }

    double getCentreOfRotationAngle(){
        return this.centreOfRotationAngle;
    }
}