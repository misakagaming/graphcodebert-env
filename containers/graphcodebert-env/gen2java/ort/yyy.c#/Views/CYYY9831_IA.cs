// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: CYYY9831_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:40:32
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
// START OF IMPORT VIEW
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
namespace GEN.ORT.YYY
{
  /// <summary>
  /// Internal data view storage for: CYYY9831_IA
  /// </summary>
  [Serializable]
  public class CYYY9831_IA : ViewBase, IImportView
  {
    private static CYYY9831_IA[] freeArray = new CYYY9831_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_ERROR
    //        Type: ISC1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentSeverityCode
    /// </summary>
    private char _ImpErrorIsc1ComponentSeverityCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentSeverityCode
    /// </summary>
    public char ImpErrorIsc1ComponentSeverityCode_AS {
      get {
        return(_ImpErrorIsc1ComponentSeverityCode_AS);
      }
      set {
        _ImpErrorIsc1ComponentSeverityCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentSeverityCode
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIsc1ComponentSeverityCode;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentSeverityCode
    /// </summary>
    public string ImpErrorIsc1ComponentSeverityCode {
      get {
        return(_ImpErrorIsc1ComponentSeverityCode);
      }
      set {
        _ImpErrorIsc1ComponentSeverityCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentRollbackIndicator
    /// </summary>
    private char _ImpErrorIsc1ComponentRollbackIndicator_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentRollbackIndicator
    /// </summary>
    public char ImpErrorIsc1ComponentRollbackIndicator_AS {
      get {
        return(_ImpErrorIsc1ComponentRollbackIndicator_AS);
      }
      set {
        _ImpErrorIsc1ComponentRollbackIndicator_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentRollbackIndicator
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIsc1ComponentRollbackIndicator;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentRollbackIndicator
    /// </summary>
    public string ImpErrorIsc1ComponentRollbackIndicator {
      get {
        return(_ImpErrorIsc1ComponentRollbackIndicator);
      }
      set {
        _ImpErrorIsc1ComponentRollbackIndicator = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentOriginServid
    /// </summary>
    private char _ImpErrorIsc1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentOriginServid
    /// </summary>
    public char ImpErrorIsc1ComponentOriginServid_AS {
      get {
        return(_ImpErrorIsc1ComponentOriginServid_AS);
      }
      set {
        _ImpErrorIsc1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _ImpErrorIsc1ComponentOriginServid;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentOriginServid
    /// </summary>
    public double ImpErrorIsc1ComponentOriginServid {
      get {
        return(_ImpErrorIsc1ComponentOriginServid);
      }
      set {
        _ImpErrorIsc1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentContextString
    /// </summary>
    private char _ImpErrorIsc1ComponentContextString_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentContextString
    /// </summary>
    public char ImpErrorIsc1ComponentContextString_AS {
      get {
        return(_ImpErrorIsc1ComponentContextString_AS);
      }
      set {
        _ImpErrorIsc1ComponentContextString_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentContextString
    /// Domain: Text
    /// Length: 512
    /// Varying Length: Y
    /// </summary>
    private string _ImpErrorIsc1ComponentContextString;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentContextString
    /// </summary>
    public string ImpErrorIsc1ComponentContextString {
      get {
        return(_ImpErrorIsc1ComponentContextString);
      }
      set {
        _ImpErrorIsc1ComponentContextString = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentReturnCode
    /// </summary>
    private char _ImpErrorIsc1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentReturnCode
    /// </summary>
    public char ImpErrorIsc1ComponentReturnCode_AS {
      get {
        return(_ImpErrorIsc1ComponentReturnCode_AS);
      }
      set {
        _ImpErrorIsc1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpErrorIsc1ComponentReturnCode;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentReturnCode
    /// </summary>
    public int ImpErrorIsc1ComponentReturnCode {
      get {
        return(_ImpErrorIsc1ComponentReturnCode);
      }
      set {
        _ImpErrorIsc1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentReasonCode
    /// </summary>
    private char _ImpErrorIsc1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentReasonCode
    /// </summary>
    public char ImpErrorIsc1ComponentReasonCode_AS {
      get {
        return(_ImpErrorIsc1ComponentReasonCode_AS);
      }
      set {
        _ImpErrorIsc1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpErrorIsc1ComponentReasonCode;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentReasonCode
    /// </summary>
    public int ImpErrorIsc1ComponentReasonCode {
      get {
        return(_ImpErrorIsc1ComponentReasonCode);
      }
      set {
        _ImpErrorIsc1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentDialectCd
    /// </summary>
    private char _ImpErrorIsc1ComponentDialectCd_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentDialectCd
    /// </summary>
    public char ImpErrorIsc1ComponentDialectCd_AS {
      get {
        return(_ImpErrorIsc1ComponentDialectCd_AS);
      }
      set {
        _ImpErrorIsc1ComponentDialectCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentDialectCd
    /// Domain: Text
    /// Length: 2
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIsc1ComponentDialectCd;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentDialectCd
    /// </summary>
    public string ImpErrorIsc1ComponentDialectCd {
      get {
        return(_ImpErrorIsc1ComponentDialectCd);
      }
      set {
        _ImpErrorIsc1ComponentDialectCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpErrorIsc1ComponentChecksum
    /// </summary>
    private char _ImpErrorIsc1ComponentChecksum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpErrorIsc1ComponentChecksum
    /// </summary>
    public char ImpErrorIsc1ComponentChecksum_AS {
      get {
        return(_ImpErrorIsc1ComponentChecksum_AS);
      }
      set {
        _ImpErrorIsc1ComponentChecksum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpErrorIsc1ComponentChecksum
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _ImpErrorIsc1ComponentChecksum;
    /// <summary>
    /// Attribute for: ImpErrorIsc1ComponentChecksum
    /// </summary>
    public string ImpErrorIsc1ComponentChecksum {
      get {
        return(_ImpErrorIsc1ComponentChecksum);
      }
      set {
        _ImpErrorIsc1ComponentChecksum = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public CYYY9831_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public CYYY9831_IA( CYYY9831_IA orig )
    {
      ImpErrorIsc1ComponentSeverityCode_AS = orig.ImpErrorIsc1ComponentSeverityCode_AS;
      ImpErrorIsc1ComponentSeverityCode = orig.ImpErrorIsc1ComponentSeverityCode;
      ImpErrorIsc1ComponentRollbackIndicator_AS = orig.ImpErrorIsc1ComponentRollbackIndicator_AS;
      ImpErrorIsc1ComponentRollbackIndicator = orig.ImpErrorIsc1ComponentRollbackIndicator;
      ImpErrorIsc1ComponentOriginServid_AS = orig.ImpErrorIsc1ComponentOriginServid_AS;
      ImpErrorIsc1ComponentOriginServid = orig.ImpErrorIsc1ComponentOriginServid;
      ImpErrorIsc1ComponentContextString_AS = orig.ImpErrorIsc1ComponentContextString_AS;
      ImpErrorIsc1ComponentContextString = orig.ImpErrorIsc1ComponentContextString;
      ImpErrorIsc1ComponentReturnCode_AS = orig.ImpErrorIsc1ComponentReturnCode_AS;
      ImpErrorIsc1ComponentReturnCode = orig.ImpErrorIsc1ComponentReturnCode;
      ImpErrorIsc1ComponentReasonCode_AS = orig.ImpErrorIsc1ComponentReasonCode_AS;
      ImpErrorIsc1ComponentReasonCode = orig.ImpErrorIsc1ComponentReasonCode;
      ImpErrorIsc1ComponentDialectCd_AS = orig.ImpErrorIsc1ComponentDialectCd_AS;
      ImpErrorIsc1ComponentDialectCd = orig.ImpErrorIsc1ComponentDialectCd;
      ImpErrorIsc1ComponentChecksum_AS = orig.ImpErrorIsc1ComponentChecksum_AS;
      ImpErrorIsc1ComponentChecksum = orig.ImpErrorIsc1ComponentChecksum;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static CYYY9831_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new CYYY9831_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new CYYY9831_IA());
          }
          else 
          {
            CYYY9831_IA result = freeArray[--countFree];
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
      return(new CYYY9831_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpErrorIsc1ComponentSeverityCode_AS = ' ';
      ImpErrorIsc1ComponentSeverityCode = " ";
      ImpErrorIsc1ComponentRollbackIndicator_AS = ' ';
      ImpErrorIsc1ComponentRollbackIndicator = " ";
      ImpErrorIsc1ComponentOriginServid_AS = ' ';
      ImpErrorIsc1ComponentOriginServid = 0.0;
      ImpErrorIsc1ComponentContextString_AS = ' ';
      ImpErrorIsc1ComponentContextString = "";
      ImpErrorIsc1ComponentReturnCode_AS = ' ';
      ImpErrorIsc1ComponentReturnCode = 0;
      ImpErrorIsc1ComponentReasonCode_AS = ' ';
      ImpErrorIsc1ComponentReasonCode = 0;
      ImpErrorIsc1ComponentDialectCd_AS = ' ';
      ImpErrorIsc1ComponentDialectCd = "  ";
      ImpErrorIsc1ComponentChecksum_AS = ' ';
      ImpErrorIsc1ComponentChecksum = "               ";
    }
    /// <summary>
    /// Gets the VDF version of the current state of the instance.
    /// </summary>
    public VDF GetVDF(  )
    {
      throw new Exception("can only execute GetVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current state of the instance to the VDF version.
    /// </summary>
    public void SetFromVDF( VDF vdf )
    {
      throw new Exception("can only execute SetFromVDF for a Procedure Step.");
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IImportView orig )
    {
      this.CopyFrom((CYYY9831_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( CYYY9831_IA orig )
    {
      ImpErrorIsc1ComponentSeverityCode_AS = orig.ImpErrorIsc1ComponentSeverityCode_AS;
      ImpErrorIsc1ComponentSeverityCode = orig.ImpErrorIsc1ComponentSeverityCode;
      ImpErrorIsc1ComponentRollbackIndicator_AS = orig.ImpErrorIsc1ComponentRollbackIndicator_AS;
      ImpErrorIsc1ComponentRollbackIndicator = orig.ImpErrorIsc1ComponentRollbackIndicator;
      ImpErrorIsc1ComponentOriginServid_AS = orig.ImpErrorIsc1ComponentOriginServid_AS;
      ImpErrorIsc1ComponentOriginServid = orig.ImpErrorIsc1ComponentOriginServid;
      ImpErrorIsc1ComponentContextString_AS = orig.ImpErrorIsc1ComponentContextString_AS;
      ImpErrorIsc1ComponentContextString = orig.ImpErrorIsc1ComponentContextString;
      ImpErrorIsc1ComponentReturnCode_AS = orig.ImpErrorIsc1ComponentReturnCode_AS;
      ImpErrorIsc1ComponentReturnCode = orig.ImpErrorIsc1ComponentReturnCode;
      ImpErrorIsc1ComponentReasonCode_AS = orig.ImpErrorIsc1ComponentReasonCode_AS;
      ImpErrorIsc1ComponentReasonCode = orig.ImpErrorIsc1ComponentReasonCode;
      ImpErrorIsc1ComponentDialectCd_AS = orig.ImpErrorIsc1ComponentDialectCd_AS;
      ImpErrorIsc1ComponentDialectCd = orig.ImpErrorIsc1ComponentDialectCd;
      ImpErrorIsc1ComponentChecksum_AS = orig.ImpErrorIsc1ComponentChecksum_AS;
      ImpErrorIsc1ComponentChecksum = orig.ImpErrorIsc1ComponentChecksum;
    }
  }
}