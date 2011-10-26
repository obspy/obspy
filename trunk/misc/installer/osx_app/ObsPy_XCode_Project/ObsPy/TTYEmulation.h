//
//  TTYEmulation.h
//
//  Created by Lion Krischer on 12.9.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface TTYEmulation : NSObject
{   
    NSFileHandle * masterFileHandler;
    NSFileHandle * slaveFileHandler;
    NSString * slaveFileName;
    NSMutableString * textString;
    NSWindow * window;
    NSTextView * textView;
    NSTask * task;
    
    NSArray * taskQueue;
    NSUInteger currentTaskIndex;
    
    // Configuration
    BOOL showWindow;
    BOOL printCommandToWindow;

}

@property (nonatomic) BOOL showWindow;
@property (nonatomic) BOOL printCommandToWindow;

-(id)initWithTextView: (NSTextView *)view andViewWindow: (NSWindow *)viewWindow;
-(void)didRead: (NSNotification *)noty;
-(void)stopTask;

-(void)runSingleCommand: (NSString *)cmd withArguments:(NSArray *)args;
-(void)runMultipleCommands: (NSArray *)commands;
-(void)runNextTask;
-(void)allTasksDone;

@end
