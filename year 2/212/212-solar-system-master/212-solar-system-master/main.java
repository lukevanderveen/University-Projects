public class main { 
    public static void main(){
        int rotating = 1;
        
        //solar system decleration
        SolarSystem system = new SolarSystem(700, 700); 
        //planet deceleration
        Planet Sun = new Planet(system, 0, 0, 50, "YELLOW", 0, 0);
        Planet Mercury = new Planet(system, 40, 0, 10, "GRAY", 2.5);
        Planet Venus = new Planet(system, 60, 0, 10, "#FFA756", 2.2);
        Planet Earth = new Planet(system, 80, 0, 20, "GREEN", 1.9);
        Planet Mars = new Planet(system, 95, 0, 15, "RED", 1.7);
        Planet Jupiter = new Planet(system, 120, 0, 35, "#FFB347", 1.65);
        Planet Saturn = new Planet(system, 160, 0, 30, "#CDAA74", 1.55);
        Planet Uranus = new Planet(system, 180, 0, 15, "CYAN", 1.45);
        Planet Neptune = new Planet(system, 200, 0, 15, "BLUE", 1.3);
        //i know it isn't a planet 
        Planet Pluto = new Planet(system, 210, 0, 5, "LIGHT GRAY", 1.2);

        //moon (earth's)

        while (rotating != 0){
            system.drawSolarObject(Sun.getDistance(), Sun.getAngle(), Sun.getDiameter(), Sun.getCol());
            system.drawSolarObject(Mercury.getDistance(), Mercury.getAngle(), Mercury.getDiameter(), Mercury.getCol());
            system.drawSolarObject(Venus.getDistance(), Venus.getAngle(), Venus.getDiameter(), Venus.getCol());
            system.drawSolarObject(Earth.getDistance(), Earth.getAngle(), Earth.getDiameter(), Earth.getCol());
            system.drawSolarObject(Mars.getDistance(), Mars.getAngle(), Mars.getDiameter(), Mars.getCol());
            system.drawSolarObject(Jupiter.getDistance(), Jupiter.getAngle(), Jupiter.getDiameter(), Jupiter.getCol());
            system.drawSolarObject(Saturn.getDistance(), Saturn.getAngle(), Saturn.getDiameter(), Saturn.getCol());
            system.drawSolarObject(Uranus.getDistance(), Uranus.getAngle(), Uranus.getDiameter(), Uranus.getCol());
            system.drawSolarObject(Neptune.getDistance(), Neptune.getAngle(), Neptune.getDiameter(), Neptune.getCol());
            system.drawSolarObject(Pluto.getDistance(), Pluto.getAngle(), Pluto.getDiameter(), Pluto.getCol());

            Mercury.move();
            Venus.move();
            Earth.move();
            Mars.move();
            Jupiter.move();
            Saturn.move();
            Uranus.move();
            Neptune.move();
            Pluto.move();


            //refresh the screen
            system.finishedDrawing();
        }        
    }

}

