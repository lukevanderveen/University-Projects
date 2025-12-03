"use strict";

const testlib = require( './testlib.js' );

let test = [];
const pattern =  ['AA', 'CC', 'TT', 'GG'];


testlib.on( 'ready', function( patterns ) {
    console.log( "Patterns:", patterns );
    testlib.runTests();
} );

testlib.on( 'data', function( data ) {
    test.push(data);
   
} );


testlib.on( 'reset', function() {
    console.log(test);
    //indexes goes through array scans for match of first pattern then scans again for next pattern etc
    let indexes = pattern.flatMap(pattern => 
        test.reduce((acc, _, index) => {
            const seq = test.slice(index, index + pattern.length).join('');//create string
            if (seq === pattern) {
                testlib.foundMatch(pattern, index);
                return [...acc, index]; //add index to next position in array (acc)
            }
            return acc;
        }, [])
    );
    console.log(indexes);
    test = [];
} );
   
testlib.on('end', function() {
});



testlib.setup( 2 );


/*

 



*/