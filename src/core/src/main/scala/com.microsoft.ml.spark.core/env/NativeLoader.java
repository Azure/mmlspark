// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark;

import java.io.*;
import java.nio.file.Files;

/**
 * A helper class for loading native libraries from Java
 *
 * <p>Some Java interfaces depend on native libraries that need to be loaded at runtime.
 * This class is a simple utility that can load the native libraries from a jar in one of two ways:</p>
 *
 * <ul>
 * <li>By name: If a particular native library is needed, it will extract it to a temp folder
 *     (along with its dependencies) and load it from there.</li>
 * <li>All libraries: all libraries will be extracted to a temp folder and the libraries in the
 *     load manifest are loaded in the order provided, or loaded in the order specified in the
 *     native manifest if no load manifest is provided. </li>
 * </ul>
 *
 * <p>The jar with the native libraries must contain a file name 'NATIVE_MANIFEST' that lists
 * all native files (one per line, full name) to be extracted. If the loadAll() method is used,
 * the libraries will be loaded in the order specified in the manifest. The native libraries should be
 * in folders describing the OS they run on: linux, windows, mac. </p>
 * */
public class NativeLoader implements Serializable {

    private String resourcesPath;
    private Boolean extractionDone = false;
    private File tempDir;

    public NativeLoader(String topLevelResourcesPath) throws IOException {
        this.resourcesPath = getResourcesPath(topLevelResourcesPath);
        tempDir = Files.createTempDirectory("mml-natives").toFile();
        tempDir.deleteOnExit();
    }

    /**
     * Loads a named native library from the jar file
     *
     * <p>This method will first try to load the library from java.library.path system property.
     * Only if that fails, the named native library and its dependencies will be extracted to
     * a temporary folder and loaded from there.</p>
     * */
    public void loadLibraryByName(String libName) {
        try {
            // First try loading by name
            // It's possible that the native library is already on a path java can discover
            System.loadLibrary(libName);
        }
        catch (UnsatisfiedLinkError e) {
            try {
                // Get the OS specific library name
                libName = System.mapLibraryName(libName);
                extractNativeLibraries(libName);
                // Try to load library from extracted native resources
                System.load(tempDir.getAbsolutePath() + File.separator + libName);
            }
            catch (Exception ee) {
                throw new UnsatisfiedLinkError(String.format(
                        "Could not load the native libraries because " +
                        "we encountered the following problems: %s and %s",
                        e.getMessage(), ee.getMessage()));
            }
        }
    }

    private void extractNativeLibraries(String libName) throws IOException {
        if (!extractionDone) {
            extractResourceFromPath(libName, resourcesPath);
        }
        extractionDone = true;
    }

    private static String getResourcesPath(String topLevelResourcesPath) {
        String sep = "/";
        String OS = System.getProperty("os.name").toLowerCase();
        String resourcePrefix = topLevelResourcesPath + sep + "%s" + sep;
        if (OS.contains("linux")) {
            return String.format(resourcePrefix, "linux/x86_64");
        } else if (OS.contains("windows")) {
            return String.format(resourcePrefix, "windows/x86_64");
        } else if (OS.contains("mac") || OS.contains("darwin")) {
            return String.format(resourcePrefix, "osx/x86_64");
        } else {
            throw new UnsatisfiedLinkError(
                String.format("This component doesn't currently have native support for OS: %s", OS)
            );
        }
    }

    private void extractResourceFromPath(String libName, String prefix) throws IOException {

        File temp = new File(tempDir.getPath() + File.separator + libName);
        temp.createNewFile();
        temp.deleteOnExit();

        if (!temp.exists()) {
            throw new FileNotFoundException(String.format(
                    "Temporary file %s could not be created. Make sure you can write to this location.",
                    temp.getAbsolutePath())
            );
        }

        String path = prefix + libName;
        InputStream inStream = NativeLoader.class.getResourceAsStream(path);
        if (inStream == null) {
            throw new FileNotFoundException(String.format("Could not find resource %s in jar.", path));
        }

        FileOutputStream outStream = new FileOutputStream(temp);
        byte[] buffer = new byte[1 << 18];
        int bytesRead;

        try {
            while ((bytesRead = inStream.read(buffer)) >= 0) {
                outStream.write(buffer, 0, bytesRead);
            }
        } finally {
            outStream.close();
            inStream.close();
        }
    }

}
