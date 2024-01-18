// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY0411_OA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:24
//
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// using Statements
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
using System;
using com.ca.gen.vwrt;
using com.ca.gen.vwrt.types;
using com.ca.gen.vwrt.vdf;
using com.ca.gen.csu.exception;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// START OF EXPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: CYYY0411_OA
  /// </summary>
  [Serializable]
  public class CYYY0411_OA : ViewBase, IExportView
  {
    private static CYYY0411_OA[] freeArray = new CYYY0411_OA[30];
    private static int countFree = 0;
    
    // Entity View: EXP
    //        Type: CANAM_XML
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpCanamXmlXmlBuffer
    /// </summary>
    private char _ExpCanamXmlXmlBuffer_AS;
    /// <summary>
    /// Attribute missing flag for: ExpCanamXmlXmlBuffer
    /// </summary>
    public char ExpCanamXmlXmlBuffer_AS {
      get {
        return(_ExpCanamXmlXmlBuffer_AS);
      }
      set {
        _ExpCanamXmlXmlBuffer_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpCanamXmlXmlBuffer
    /// Domain: Text
    /// Length: 4094
    /// Varying Length: Y
    /// </summary>
    private string _ExpCanamXmlXmlBuffer;
    /// <summary>
    /// Attribute for: ExpCanamXmlXmlBuffer
    /// </summary>
    public string ExpCanamXmlXmlBuffer {
      get {
        return(_ExpCanamXmlXmlBuffer);
      }
      set {
        _ExpCanamXmlXmlBuffer = value;
      }
    }
    // Entity View: EXP_ERROR
    //        Type: CANAM_XML
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorCanamXmlXmlReturnCode
    /// </summary>
    private char _ExpErrorCanamXmlXmlReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorCanamXmlXmlReturnCode
    /// </summary>
    public char ExpErrorCanamXmlXmlReturnCode_AS {
      get {
        return(_ExpErrorCanamXmlXmlReturnCode_AS);
      }
      set {
        _ExpErrorCanamXmlXmlReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorCanamXmlXmlReturnCode
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorCanamXmlXmlReturnCode;
    /// <summary>
    /// Attribute for: ExpErrorCanamXmlXmlReturnCode
    /// </summary>
    public string ExpErrorCanamXmlXmlReturnCode {
      get {
        return(_ExpErrorCanamXmlXmlReturnCode);
      }
      set {
        _ExpErrorCanamXmlXmlReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorCanamXmlXmlMessage
    /// </summary>
    private char _ExpErrorCanamXmlXmlMessage_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorCanamXmlXmlMessage
    /// </summary>
    public char ExpErrorCanamXmlXmlMessage_AS {
      get {
        return(_ExpErrorCanamXmlXmlMessage_AS);
      }
      set {
        _ExpErrorCanamXmlXmlMessage_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorCanamXmlXmlMessage
    /// Domain: Text
    /// Length: 80
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorCanamXmlXmlMessage;
    /// <summary>
    /// Attribute for: ExpErrorCanamXmlXmlMessage
    /// </summary>
    public string ExpErrorCanamXmlXmlMessage {
      get {
        return(_ExpErrorCanamXmlXmlMessage);
      }
      set {
        _ExpErrorCanamXmlXmlMessage = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorCanamXmlXmlPosition
    /// </summary>
    private char _ExpErrorCanamXmlXmlPosition_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorCanamXmlXmlPosition
    /// </summary>
    public char ExpErrorCanamXmlXmlPosition_AS {
      get {
        return(_ExpErrorCanamXmlXmlPosition_AS);
      }
      set {
        _ExpErrorCanamXmlXmlPosition_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorCanamXmlXmlPosition
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ExpErrorCanamXmlXmlPosition;
    /// <summary>
    /// Attribute for: ExpErrorCanamXmlXmlPosition
    /// </summary>
    public double ExpErrorCanamXmlXmlPosition {
      get {
        return(_ExpErrorCanamXmlXmlPosition);
      }
      set {
        _ExpErrorCanamXmlXmlPosition = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorCanamXmlXmlSource
    /// </summary>
    private char _ExpErrorCanamXmlXmlSource_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorCanamXmlXmlSource
    /// </summary>
    public char ExpErrorCanamXmlXmlSource_AS {
      get {
        return(_ExpErrorCanamXmlXmlSource_AS);
      }
      set {
        _ExpErrorCanamXmlXmlSource_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorCanamXmlXmlSource
    /// Domain: Text
    /// Length: 120
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorCanamXmlXmlSource;
    /// <summary>
    /// Attribute for: ExpErrorCanamXmlXmlSource
    /// </summary>
    public string ExpErrorCanamXmlXmlSource {
      get {
        return(_ExpErrorCanamXmlXmlSource);
      }
      set {
        _ExpErrorCanamXmlXmlSource = value;
      }
    }
    // Entity View: EXP_ERROR
    //        Type: IYY1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    private char _ExpErrorIyy1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public char ExpErrorIyy1ComponentSeverityCode_AS {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentSeverityCode
    /// </summary>
    public string ExpErrorIyy1ComponentSeverityCode {
      get {
        return(_ExpErrorIyy1ComponentSeverityCode);
      }
      set {
        _ExpErrorIyy1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    private char _ExpErrorIyy1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public char ExpErrorIyy1ComponentRollbackIndicator_AS {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator_AS);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentRollbackIndicator
    /// </summary>
    public string ExpErrorIyy1ComponentRollbackIndicator {
      get {
        return(_ExpErrorIyy1ComponentRollbackIndicator);
      }
      set {
        _ExpErrorIyy1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    private char _ExpErrorIyy1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public char ExpErrorIyy1ComponentOriginServid_AS {
      get {
        return(_ExpErrorIyy1ComponentOriginServid_AS);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ExpErrorIyy1ComponentOriginServid;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentOriginServid
    /// </summary>
    public double ExpErrorIyy1ComponentOriginServid {
      get {
        return(_ExpErrorIyy1ComponentOriginServid);
      }
      set {
        _ExpErrorIyy1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    private char _ExpErrorIyy1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public char ExpErrorIyy1ComponentContextString_AS {
      get {
        return(_ExpErrorIyy1ComponentContextString_AS);
      }
      set {
        _ExpErrorIyy1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _ExpErrorIyy1ComponentContextString;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentContextString
    /// </summary>
    public string ExpErrorIyy1ComponentContextString {
      get {
        return(_ExpErrorIyy1ComponentContextString);
      }
      set {
        _ExpErrorIyy1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public char ExpErrorIyy1ComponentReturnCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReturnCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReturnCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReturnCode
    /// </summary>
    public int ExpErrorIyy1ComponentReturnCode {
      get {
        return(_ExpErrorIyy1ComponentReturnCode);
      }
      set {
        _ExpErrorIyy1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    private char _ExpErrorIyy1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public char ExpErrorIyy1ComponentReasonCode_AS {
      get {
        return(_ExpErrorIyy1ComponentReasonCode_AS);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ExpErrorIyy1ComponentReasonCode;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentReasonCode
    /// </summary>
    public int ExpErrorIyy1ComponentReasonCode {
      get {
        return(_ExpErrorIyy1ComponentReasonCode);
      }
      set {
        _ExpErrorIyy1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    private char _ExpErrorIyy1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public char ExpErrorIyy1ComponentChecksum_AS {
      get {
        return(_ExpErrorIyy1ComponentChecksum_AS);
      }
      set {
        _ExpErrorIyy1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ExpErrorIyy1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _ExpErrorIyy1ComponentChecksum;
    /// <summary>
    /// Attribute for: ExpErrorIyy1ComponentChecksum
    /// </summary>
    public string ExpErrorIyy1ComponentChecksum {
      get {
        return(_ExpErrorIyy1ComponentChecksum);
      }
      set {
        _ExpErrorIyy1ComponentChecksum = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY0411_OA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY0411_OA( CYYY0411_OA orig )
    {
      ExpCanamXmlXmlBuffer_AS = orig.ExpCanamXmlXmlBuffer_AS;
      ExpCanamXmlXmlBuffer = orig.ExpCanamXmlXmlBuffer;
      ExpErrorCanamXmlXmlReturnCode_AS = orig.ExpErrorCanamXmlXmlReturnCode_AS;
      ExpErrorCanamXmlXmlReturnCode = orig.ExpErrorCanamXmlXmlReturnCode;
      ExpErrorCanamXmlXmlMessage_AS = orig.ExpErrorCanamXmlXmlMessage_AS;
      ExpErrorCanamXmlXmlMessage = orig.ExpErrorCanamXmlXmlMessage;
      ExpErrorCanamXmlXmlPosition_AS = orig.ExpErrorCanamXmlXmlPosition_AS;
      ExpErrorCanamXmlXmlPosition = orig.ExpErrorCanamXmlXmlPosition;
      ExpErrorCanamXmlXmlSource_AS = orig.ExpErrorCanamXmlXmlSource_AS;
      ExpErrorCanamXmlXmlSource = orig.ExpErrorCanamXmlXmlSource;
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY0411_OA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY0411_OA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY0411_OA());
          }
          else 
          {
            CYYY0411_OA result = freeArray[--countFree];
            freeArray[countFree] = null;
            result.Reset(  );
            return(result);
          }
        }
      }
    }
    /// <summary>
    /// Static free instance method
    /// </summary>
    
    public void FreeInstance(  )
    {
      lock (freeArray)
      {
        if ( countFree < freeArray.Length )
        {
          freeArray[countFree++] = this;
        }
      }
    }
    /// <summary>
    /// clone constructor
    /// </summary>
    
    public override Object Clone(  )
    {
      return(new CYYY0411_OA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ExpCanamXmlXmlBuffer_AS = ' ';
      ExpCanamXmlXmlBuffer = "";
      ExpErrorCanamXmlXmlReturnCode_AS = ' ';
      ExpErrorCanamXmlXmlReturnCode = "  ";
      ExpErrorCanamXmlXmlMessage_AS = ' ';
      ExpErrorCanamXmlXmlMessage = "                                                                                ";
      ExpErrorCanamXmlXmlPosition_AS = ' ';
      ExpErrorCanamXmlXmlPosition = 0.0;
      ExpErrorCanamXmlXmlSource_AS = ' ';
      ExpErrorCanamXmlXmlSource = 
        "                                                                                                                        ";
      ExpErrorIyy1ComponentSeverityCode_AS = ' ';
      ExpErrorIyy1ComponentSeverityCode = " ";
      ExpErrorIyy1ComponentRollbackIndicator_AS = ' ';
      ExpErrorIyy1ComponentRollbackIndicator = " ";
      ExpErrorIyy1ComponentOriginServid_AS = ' ';
      ExpErrorIyy1ComponentOriginServid = 0.0;
      ExpErrorIyy1ComponentContextString_AS = ' ';
      ExpErrorIyy1ComponentContextString = "";
      ExpErrorIyy1ComponentReturnCode_AS = ' ';
      ExpErrorIyy1ComponentReturnCode = 0;
      ExpErrorIyy1ComponentReasonCode_AS = ' ';
      ExpErrorIyy1ComponentReasonCode = 0;
      ExpErrorIyy1ComponentChecksum_AS = ' ';
      ExpErrorIyy1ComponentChecksum = "               ";
    }
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IExportView orig )
    {
      this.CopyFrom((CYYY0411_OA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( CYYY0411_OA orig )
    {
      ExpCanamXmlXmlBuffer_AS = orig.ExpCanamXmlXmlBuffer_AS;
      ExpCanamXmlXmlBuffer = orig.ExpCanamXmlXmlBuffer;
      ExpErrorCanamXmlXmlReturnCode_AS = orig.ExpErrorCanamXmlXmlReturnCode_AS;
      ExpErrorCanamXmlXmlReturnCode = orig.ExpErrorCanamXmlXmlReturnCode;
      ExpErrorCanamXmlXmlMessage_AS = orig.ExpErrorCanamXmlXmlMessage_AS;
      ExpErrorCanamXmlXmlMessage = orig.ExpErrorCanamXmlXmlMessage;
      ExpErrorCanamXmlXmlPosition_AS = orig.ExpErrorCanamXmlXmlPosition_AS;
      ExpErrorCanamXmlXmlPosition = orig.ExpErrorCanamXmlXmlPosition;
      ExpErrorCanamXmlXmlSource_AS = orig.ExpErrorCanamXmlXmlSource_AS;
      ExpErrorCanamXmlXmlSource = orig.ExpErrorCanamXmlXmlSource;
      ExpErrorIyy1ComponentSeverityCode_AS = orig.ExpErrorIyy1ComponentSeverityCode_AS;
      ExpErrorIyy1ComponentSeverityCode = orig.ExpErrorIyy1ComponentSeverityCode;
      ExpErrorIyy1ComponentRollbackIndicator_AS = orig.ExpErrorIyy1ComponentRollbackIndicator_AS;
      ExpErrorIyy1ComponentRollbackIndicator = orig.ExpErrorIyy1ComponentRollbackIndicator;
      ExpErrorIyy1ComponentOriginServid_AS = orig.ExpErrorIyy1ComponentOriginServid_AS;
      ExpErrorIyy1ComponentOriginServid = orig.ExpErrorIyy1ComponentOriginServid;
      ExpErrorIyy1ComponentContextString_AS = orig.ExpErrorIyy1ComponentContextString_AS;
      ExpErrorIyy1ComponentContextString = orig.ExpErrorIyy1ComponentContextString;
      ExpErrorIyy1ComponentReturnCode_AS = orig.ExpErrorIyy1ComponentReturnCode_AS;
      ExpErrorIyy1ComponentReturnCode = orig.ExpErrorIyy1ComponentReturnCode;
      ExpErrorIyy1ComponentReasonCode_AS = orig.ExpErrorIyy1ComponentReasonCode_AS;
      ExpErrorIyy1ComponentReasonCode = orig.ExpErrorIyy1ComponentReasonCode;
      ExpErrorIyy1ComponentChecksum_AS = orig.ExpErrorIyy1ComponentChecksum_AS;
      ExpErrorIyy1ComponentChecksum = orig.ExpErrorIyy1ComponentChecksum;
    }
  }
}
