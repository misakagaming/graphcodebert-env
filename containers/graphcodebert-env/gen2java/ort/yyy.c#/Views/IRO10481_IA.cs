// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: IRO10481_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:41:48
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
  /// Internal data view storage for: IRO10481_IA
  /// </summary>
  [Serializable]
  public class IRO10481_IA : ViewBase, IImportView
  {
    private static IRO10481_IA[] freeArray = new IRO10481_IA[30];
    private static int countFree = 0;
    
    // Entity View: IN
    //        Type: IRO1_COMPONENT
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentReturnCode
    /// </summary>
    private char _InIro1ComponentReturnCode_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentReturnCode
    /// </summary>
    public char InIro1ComponentReturnCode_AS {
      get {
        return(_InIro1ComponentReturnCode_AS);
      }
      set {
        _InIro1ComponentReturnCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentReturnCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _InIro1ComponentReturnCode;
    /// <summary>
    /// Attribute for: InIro1ComponentReturnCode
    /// </summary>
    public int InIro1ComponentReturnCode {
      get {
        return(_InIro1ComponentReturnCode);
      }
      set {
        _InIro1ComponentReturnCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentReasonCode
    /// </summary>
    private char _InIro1ComponentReasonCode_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentReasonCode
    /// </summary>
    public char InIro1ComponentReasonCode_AS {
      get {
        return(_InIro1ComponentReasonCode_AS);
      }
      set {
        _InIro1ComponentReasonCode_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentReasonCode
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _InIro1ComponentReasonCode;
    /// <summary>
    /// Attribute for: InIro1ComponentReasonCode
    /// </summary>
    public int InIro1ComponentReasonCode {
      get {
        return(_InIro1ComponentReasonCode);
      }
      set {
        _InIro1ComponentReasonCode = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentActivityCd
    /// </summary>
    private char _InIro1ComponentActivityCd_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentActivityCd
    /// </summary>
    public char InIro1ComponentActivityCd_AS {
      get {
        return(_InIro1ComponentActivityCd_AS);
      }
      set {
        _InIro1ComponentActivityCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentActivityCd
    /// Domain: Text
    /// Length: 15
    /// Varying Length: N
    /// </summary>
    private string _InIro1ComponentActivityCd;
    /// <summary>
    /// Attribute for: InIro1ComponentActivityCd
    /// </summary>
    public string InIro1ComponentActivityCd {
      get {
        return(_InIro1ComponentActivityCd);
      }
      set {
        _InIro1ComponentActivityCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentContextStringTx
    /// </summary>
    private char _InIro1ComponentContextStringTx_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentContextStringTx
    /// </summary>
    public char InIro1ComponentContextStringTx_AS {
      get {
        return(_InIro1ComponentContextStringTx_AS);
      }
      set {
        _InIro1ComponentContextStringTx_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentContextStringTx
    /// Domain: Text
    /// Length: 255
    /// Varying Length: Y
    /// </summary>
    private string _InIro1ComponentContextStringTx;
    /// <summary>
    /// Attribute for: InIro1ComponentContextStringTx
    /// </summary>
    public string InIro1ComponentContextStringTx {
      get {
        return(_InIro1ComponentContextStringTx);
      }
      set {
        _InIro1ComponentContextStringTx = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentDialectCd
    /// </summary>
    private char _InIro1ComponentDialectCd_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentDialectCd
    /// </summary>
    public char InIro1ComponentDialectCd_AS {
      get {
        return(_InIro1ComponentDialectCd_AS);
      }
      set {
        _InIro1ComponentDialectCd_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentDialectCd
    /// Domain: Text
    /// Length: 8
    /// Varying Length: N
    /// </summary>
    private string _InIro1ComponentDialectCd;
    /// <summary>
    /// Attribute for: InIro1ComponentDialectCd
    /// </summary>
    public string InIro1ComponentDialectCd {
      get {
        return(_InIro1ComponentDialectCd);
      }
      set {
        _InIro1ComponentDialectCd = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentSpecificationId
    /// </summary>
    private char _InIro1ComponentSpecificationId_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentSpecificationId
    /// </summary>
    public char InIro1ComponentSpecificationId_AS {
      get {
        return(_InIro1ComponentSpecificationId_AS);
      }
      set {
        _InIro1ComponentSpecificationId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentSpecificationId
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _InIro1ComponentSpecificationId;
    /// <summary>
    /// Attribute for: InIro1ComponentSpecificationId
    /// </summary>
    public double InIro1ComponentSpecificationId {
      get {
        return(_InIro1ComponentSpecificationId);
      }
      set {
        _InIro1ComponentSpecificationId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentImplementationId
    /// </summary>
    private char _InIro1ComponentImplementationId_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentImplementationId
    /// </summary>
    public char InIro1ComponentImplementationId_AS {
      get {
        return(_InIro1ComponentImplementationId_AS);
      }
      set {
        _InIro1ComponentImplementationId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentImplementationId
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _InIro1ComponentImplementationId;
    /// <summary>
    /// Attribute for: InIro1ComponentImplementationId
    /// </summary>
    public double InIro1ComponentImplementationId {
      get {
        return(_InIro1ComponentImplementationId);
      }
      set {
        _InIro1ComponentImplementationId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: InIro1ComponentOriginServid
    /// </summary>
    private char _InIro1ComponentOriginServid_AS;
    /// <summary>
    /// Attribute missing flag for: InIro1ComponentOriginServid
    /// </summary>
    public char InIro1ComponentOriginServid_AS {
      get {
        return(_InIro1ComponentOriginServid_AS);
      }
      set {
        _InIro1ComponentOriginServid_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: InIro1ComponentOriginServid
    /// Domain: Number
    /// Length: 15
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private double _InIro1ComponentOriginServid;
    /// <summary>
    /// Attribute for: InIro1ComponentOriginServid
    /// </summary>
    public double InIro1ComponentOriginServid {
      get {
        return(_InIro1ComponentOriginServid);
      }
      set {
        _InIro1ComponentOriginServid = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public IRO10481_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public IRO10481_IA( IRO10481_IA orig )
    {
      InIro1ComponentReturnCode_AS = orig.InIro1ComponentReturnCode_AS;
      InIro1ComponentReturnCode = orig.InIro1ComponentReturnCode;
      InIro1ComponentReasonCode_AS = orig.InIro1ComponentReasonCode_AS;
      InIro1ComponentReasonCode = orig.InIro1ComponentReasonCode;
      InIro1ComponentActivityCd_AS = orig.InIro1ComponentActivityCd_AS;
      InIro1ComponentActivityCd = orig.InIro1ComponentActivityCd;
      InIro1ComponentContextStringTx_AS = orig.InIro1ComponentContextStringTx_AS;
      InIro1ComponentContextStringTx = orig.InIro1ComponentContextStringTx;
      InIro1ComponentDialectCd_AS = orig.InIro1ComponentDialectCd_AS;
      InIro1ComponentDialectCd = orig.InIro1ComponentDialectCd;
      InIro1ComponentSpecificationId_AS = orig.InIro1ComponentSpecificationId_AS;
      InIro1ComponentSpecificationId = orig.InIro1ComponentSpecificationId;
      InIro1ComponentImplementationId_AS = orig.InIro1ComponentImplementationId_AS;
      InIro1ComponentImplementationId = orig.InIro1ComponentImplementationId;
      InIro1ComponentOriginServid_AS = orig.InIro1ComponentOriginServid_AS;
      InIro1ComponentOriginServid = orig.InIro1ComponentOriginServid;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static IRO10481_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new IRO10481_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new IRO10481_IA());
          }
          else 
          {
            IRO10481_IA result = freeArray[--countFree];
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
      return(new IRO10481_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      InIro1ComponentReturnCode_AS = ' ';
      InIro1ComponentReturnCode = 0;
      InIro1ComponentReasonCode_AS = ' ';
      InIro1ComponentReasonCode = 0;
      InIro1ComponentActivityCd_AS = ' ';
      InIro1ComponentActivityCd = "               ";
      InIro1ComponentContextStringTx_AS = ' ';
      InIro1ComponentContextStringTx = "";
      InIro1ComponentDialectCd_AS = ' ';
      InIro1ComponentDialectCd = "        ";
      InIro1ComponentSpecificationId_AS = ' ';
      InIro1ComponentSpecificationId = 0.0;
      InIro1ComponentImplementationId_AS = ' ';
      InIro1ComponentImplementationId = 0.0;
      InIro1ComponentOriginServid_AS = ' ';
      InIro1ComponentOriginServid = 0.0;
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
      this.CopyFrom((IRO10481_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( IRO10481_IA orig )
    {
      InIro1ComponentReturnCode_AS = orig.InIro1ComponentReturnCode_AS;
      InIro1ComponentReturnCode = orig.InIro1ComponentReturnCode;
      InIro1ComponentReasonCode_AS = orig.InIro1ComponentReasonCode_AS;
      InIro1ComponentReasonCode = orig.InIro1ComponentReasonCode;
      InIro1ComponentActivityCd_AS = orig.InIro1ComponentActivityCd_AS;
      InIro1ComponentActivityCd = orig.InIro1ComponentActivityCd;
      InIro1ComponentContextStringTx_AS = orig.InIro1ComponentContextStringTx_AS;
      InIro1ComponentContextStringTx = orig.InIro1ComponentContextStringTx;
      InIro1ComponentDialectCd_AS = orig.InIro1ComponentDialectCd_AS;
      InIro1ComponentDialectCd = orig.InIro1ComponentDialectCd;
      InIro1ComponentSpecificationId_AS = orig.InIro1ComponentSpecificationId_AS;
      InIro1ComponentSpecificationId = orig.InIro1ComponentSpecificationId;
      InIro1ComponentImplementationId_AS = orig.InIro1ComponentImplementationId_AS;
      InIro1ComponentImplementationId = orig.InIro1ComponentImplementationId;
      InIro1ComponentOriginServid_AS = orig.InIro1ComponentOriginServid_AS;
      InIro1ComponentOriginServid = orig.InIro1ComponentOriginServid;
    }
  }
}