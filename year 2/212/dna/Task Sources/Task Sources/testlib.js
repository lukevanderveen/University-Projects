"use strict";

const fs = require( 'fs' );
let interval = 25;
let input_buffer = [];
let hooks = {
	ready: noop,
	data: noop,
	reset: noop,
	end: noop
};

/**
 * A no-operation function, placholder for each hook
 */
function noop(){
	// Do nothing
};

/**
 * A function to hook specific events.
 *
 * Expects two parameters:
 *     A hook string
 *     A function to handle this hook
 *
 * Valid hooks are:
 *     'ready' - handler is called with a list of sequences to match
 *     'data' - handler is called repeatedly with each input character
 *     'reset' - handler is called whenever a new input is started
 *     'end' - handler is called once when all data is complete
 */
module.exports.on = function( hook, handler ) {
	hooks[hook] = handler;
};

/**
 * Should be called when a sequence has been processed with a frequency table as an argument
 *
 * The table format is an Object where keys are the matched sequences, and their values are
 * the number of matches found.
 *
 * Example:
 * {
 *    "AA": 1,
 *    "CC": 5
 * }
 */
module.exports.frequencyTable = function( table ) {
	for( const key in table ) {
		console.log( key, table[key] );
	}
};

/**
 * Should be called for each match found.
 *
 * Should be supplied with two parameters, the first the pattern as a string, the second the
 * offset, in bytes where this match was found.
 */
module.exports.foundMatch = function( pattern, offset ) {
	console.log( "[MATCH]", pattern, "found at", offset );
};

/**
 * Requests that data begin being processed.
 *
 * Must be called after setup().
 */
module.exports.runTests = function() {
	sendByte();
};

/**
 * Sets up the library to run a set of tests.
 *
 * Accepts two arguments:
 *   - testNumber [required] - The test number to run,
 *   - ioDelay [optional] - How long to wait between data events, in milliseconds. Defaults to 25ms
 *
 * Must be called before runTests().
 */
module.exports.setup = function( testNumber, ioDelay = 25 ) {
	interval = ioDelay;
	let dataFile = fs.createReadStream( `task${testNumber}.data`, { encoding: 'utf8', fd: null } );
	dataFile.on('readable', function() {
		let chunk;
		while (null !== (chunk = dataFile.read(1))) {
			input_buffer.push( chunk[0] );
		}
	});
	dataFile.on( 'end', function() {
		fs.readFile( `task${testNumber}.seq`, function( err, data ) {
			hooks['reset']();
			hooks['ready']( data.toString().split('\n') );
		} );
	} );
}

let index = 0;
function sendByte() {
	if( input_buffer[index].charAt(0) === '\n' )
		hooks['reset']();
	else
		hooks['data']( input_buffer[index].charAt(0) );
	index++;

	if( index >= input_buffer.length ) {
		hooks['end']();
	}
	else
		setTimeout( sendByte, interval );
}
