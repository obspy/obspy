//
//  ApplicationDataModel.h
//  ObsPy
//
//  Just a simple class that serves as a data model for whatever this
//  Application needs.
//
//  Created by Lion Krischer on 8.9.11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface ApplicationDataModel : NSObject {
    NSString * applicationVersion;
    NSMutableString * promptName;
    // Virtual environment configuration.
    NSURL * virtualEnvPath;
    NSNumber * virtualEnvAddToBashProfile;
    NSNumber * virtualEnvInstallIPython;
    NSNumber * virtualEnvInstallReadline;
    NSNumber * virtualEnvUpgradeDistribute;
}

@property (readwrite, copy) NSString * applicationVersion;
@property (readwrite, copy) NSMutableString * promptName;
@property (readwrite, copy) NSURL * virtualEnvPath;
@property (readwrite, copy) NSNumber * virtualEnvAddToBashProfile;
@property (readwrite, copy) NSNumber * virtualEnvInstallIPython;
@property (readwrite, copy) NSNumber * virtualEnvInstallReadline;
@property (readwrite, copy) NSNumber * virtualEnvUpgradeDistribute;


@end


