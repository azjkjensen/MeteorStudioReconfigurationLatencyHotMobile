package service

import android.app.PendingIntent.getActivity
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.os.Process
import android.util.Log
import com.msrs.pose_estimation.NativeCallMethods
import com.msrs.pose_estimation.R
import org.opencv.core.MatOfPoint3f
import java.io.File
import java.io.FileOutputStream
import java.util.*


class IoService : Service() {

    lateinit var mReferenceImage: File

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d("IoService", "Starting service...")
        android.os.Debug.waitForDebugger();  // this line is key
        return super.onStartCommand(intent, flags, startId)
    }

    override fun onBind(intent: Intent): IBinder = binder

    fun getTime():String = "10:46"

    private val binder = object:IIoService.Stub(){
        override fun basicTypes(anInt: Int, aLong: Long, aBoolean: Boolean, aFloat: Float, aDouble: Double, aString: String?) {
            // Nothing
        }

        override fun getPid(): Int = Process.myPid()

        override fun getReferenceImage():ByteArray{

            //load the reference image
            try {
                val iStream = resources.openRawResource(R.raw.stones)
                val cascadeDir = getDir("ref", Context.MODE_PRIVATE)

                mReferenceImage = File(cascadeDir, "referenceImage.jpg")
                val os = FileOutputStream(mReferenceImage)

                val buffer = ByteArray(4096)
                var bytesRead: Int = iStream.read(buffer)
                while (bytesRead != -1) {
                    os.write(buffer, 0, bytesRead)
                    bytesRead = iStream.read(buffer)
                }

                iStream.close()
                os.close()

            } catch (e: Exception) {
                e.printStackTrace()
            }

//            val m = MatOfPoint3f()
            val ba:ByteArray = NativeCallMethods.generateReferenceImage(mReferenceImage.absolutePath)
//            val ba = FloatArray((m.total() * m.elemSize()).toInt())
//            m.get(0,0, ba)
            return ba
        }

        override fun getKeypoints():ByteArray{
            return NativeCallMethods.generateKeypointsReference()
        }

        override fun getDescriptors():ByteArray{
            return NativeCallMethods.generateDescriptorsReference()
        }
    }
}
