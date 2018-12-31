// IIoService.aidl
package service;

// Declare any non-default types here with import statements

interface IIoService {
    /**
     * Demonstrates some basic types that you can use as parameters
     * and return values in AIDL.
     */
    void basicTypes(int anInt, long aLong, boolean aBoolean, float aFloat,
            double aDouble, String aString);

    /** Request the process ID of this service, to do evil things with it. */
    int getPid();

    byte[] getReferenceImage();

    byte[] getKeypoints();
    byte[] getDescriptors();
}
