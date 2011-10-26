//
//  TTYEmulation.m
//
//  Created by Lion Krischer on 12.9.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//
#import "TTYEmulation.h"
#import <util.h>
//#include <sys/ioctl.h>
//#include <unistd.h>

@implementation TTYEmulation

@synthesize showWindow;
@synthesize printCommandToWindow;


-(id)init
{
    self = [super init];
    if (self)
    {
        // Setup file handlers and stuff.
        int c_masterFileHandler, c_slaveFileHandler;
        char c_slaveFilename[64];
        // Definition:
        // int openpty(int *amaster, int *aslave, char *name, struct termios *termp, struct winsize *winp);
        // Returns master and slave file handlers and the filename of the slave.
        if (openpty(&c_masterFileHandler, &c_slaveFileHandler, c_slaveFilename, NULL, NULL) == -1)
        {
            [NSException raise:@"TTYException" format:@"Problem with openpty: %s", strerror(errno)];
        }
        // Create Objective-C types.
        masterFileHandler = [[NSFileHandle alloc] initWithFileDescriptor:c_masterFileHandler closeOnDealloc:YES];
        slaveFileHandler = [[NSFileHandle alloc] initWithFileDescriptor:c_slaveFileHandler];
        slaveFileName = [[NSString alloc] initWithCString:c_slaveFilename];
        
        [[NSNotificationCenter defaultCenter]
         addObserver:self
         selector:@selector(didRead:)
         name:NSFileHandleReadCompletionNotification
         object:masterFileHandler];
        [masterFileHandler readInBackgroundAndNotify];
        
        // Show the output window by default
        showWindow = YES;
        // Print the command by default
        printCommandToWindow = YES;
    }
    return self;
}

-(id)initWithTextView: (NSTextView *)view andViewWindow: (NSWindow *)viewWindow
{
    [self init];
    textView = view;
    window = viewWindow;
    
    return self;
}


-(void)runSingleCommand:(NSString *)cmd withArguments:(NSArray *)args
{
    // Just pass to the multiple commands method.
    [self runMultipleCommands: [NSArray arrayWithObject: [NSArray arrayWithObjects: cmd, args, nil ]]];

}

-(void)runMultipleCommands:(NSArray *)commands
{
    [(NSButton *) [[NSApp delegate] terminalEmulationWindowCloseButton] setEnabled:NO];
    [(NSButton *) [[NSApp delegate] terminalEmulationWindowStopButton] setEnabled:YES];
    [[[NSApp delegate] terminalRunning] startAnimation:self];
    
    // Clear the NSTextView and show the window containing the NSTextView.
    [textView setString:@""];
    
    if (showWindow) {
        //[window makeKeyAndOrderFront:self];
        [NSApp beginSheet:window modalForWindow:[[NSApp delegate] window] modalDelegate:[NSApp delegate] didEndSelector:NULL contextInfo:nil];
    }
    
    // Prepare queue and start it.
    taskQueue = commands;
    [taskQueue retain];
    currentTaskIndex = 0;
    [self runNextTask];
}

-(void)runNextTask
{
    if (currentTaskIndex >= [taskQueue count]) 
    {
        [self allTasksDone];
        return;
    }

    
    NSArray * currentTask = [taskQueue objectAtIndex:currentTaskIndex];
    
    
    NSString * cmd = [currentTask objectAtIndex:0];
    NSArray * args = [currentTask objectAtIndex:1];
    
    // Print a line stating the command
    if (printCommandToWindow) {
        NSString * cmdString;
        cmdString = [NSString stringWithFormat:@"> %@ %@", cmd, [args componentsJoinedByString:@" "]];
        if (currentTaskIndex) {
            cmdString = [NSString stringWithFormat:@"\n%@", cmdString];
        }
        // Reading the font from NSTextStorage did not work for some reason so the font is hardcoded.
        NSFont * oldFont;
        NSFont * font;
        oldFont = [NSFont fontWithName:@"Helvetica" size:12.0];
        font = [[NSFontManager sharedFontManager] convertFont:oldFont toHaveTrait:NSBoldFontMask];
        // Write bold text and than the newline command in black so that all the rest is printed normally.
        NSDictionary *attributes = [NSDictionary dictionaryWithObject:font forKey:NSFontAttributeName];
        NSAttributedString *attributedString = [[[NSAttributedString alloc] initWithString:cmdString attributes:attributes] autorelease];
        [[textView textStorage] appendAttributedString:attributedString];
        attributes = [NSDictionary dictionaryWithObject:oldFont forKey:NSFontAttributeName];
        attributedString = [[[NSAttributedString alloc] initWithString:@"\n" attributes:attributes] autorelease];
        [[textView textStorage] appendAttributedString:attributedString];
    }
    
    currentTaskIndex += 1;
    
    // Create task and set input and output to the specified file handlers.
    task = [[NSTask alloc] init];
    [task setStandardOutput: slaveFileHandler];
    [task setStandardError: slaveFileHandler];
    
    [task setLaunchPath: cmd];
    [task setArguments: args];
    
    [task setCurrentDirectoryPath:[@"~" stringByExpandingTildeInPath]];
    
    
    
    [[NSNotificationCenter defaultCenter]
     addObserver:self
     selector:@selector(taskDone:)
     name:NSTaskDidTerminateNotification
     object:task];

    [task launch];
}

-(void)stopTask
{
    [task terminate];
}


-(void)taskDone: (NSNotification *) notification
{
    // Start the next task with a slight delay to give the async task execution some time to catch up and actually display the output.
    [self performSelector:@selector(runNextTask) withObject:nil afterDelay:0.1];
}

-(void)allTasksDone
{
    [(NSButton *) [[NSApp delegate] terminalEmulationWindowCloseButton] setEnabled:YES];
    [(NSButton *) [[NSApp delegate] terminalEmulationWindowStopButton] setEnabled:NO];
    [[[NSApp delegate] terminalRunning] stopAnimation:self];
}


-(void) didRead: (NSNotification *)notification
{
    NSData * data = [[notification userInfo] objectForKey:NSFileHandleNotificationDataItem];
        
    if ([data length] == 0)
        return; // end of file
        
    NSString * str = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    // Append to the end of the textStorage and scroll to the end.
    [[[textView textStorage] mutableString] appendString: str];
    NSRange range;
    range = NSMakeRange ([[textView string] length], 0);
    [textView scrollRangeToVisible: range];
    [str release];
             
    [[notification object] readInBackgroundAndNotify];

}

@end
