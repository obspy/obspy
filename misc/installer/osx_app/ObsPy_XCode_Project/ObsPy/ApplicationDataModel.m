//
//  ApplicationDataModel.m
//  ObsPy
//
//  Created by Lion Krischer on 8.9.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "ApplicationDataModel.h"

@implementation ApplicationDataModel

- (id)init
{
    self = [super init];
    if (self) {
        // Default virtual environment configuration.
        promptName = [[NSMutableString alloc] init];
        [promptName setString:@"(obspy)"];
        NSString * homePath;
        homePath = @"~";
        virtualEnvPath = [[NSURL alloc] initFileURLWithPath:[homePath stringByExpandingTildeInPath] isDirectory:YES];
        [homePath release];
        virtualEnvAddToBashProfile = [NSNumber numberWithInt:1];
        virtualEnvInstallIPython = [NSNumber numberWithInt:1];
        virtualEnvInstallReadline = [NSNumber numberWithInt:1];
        virtualEnvUpgradeDistribute = [NSNumber numberWithInt:1];
    }
    
    return self;
}

@synthesize applicationVersion;
@synthesize promptName;
@synthesize virtualEnvPath;
@synthesize virtualEnvAddToBashProfile;
@synthesize virtualEnvInstallIPython;
@synthesize virtualEnvInstallReadline;
@synthesize virtualEnvUpgradeDistribute;

@end
