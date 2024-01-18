// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
//
//                    Source Code Generated by
//                           CA Gen 8.6
//
//    Copyright (c) 2024 CA Technologies. All rights reserved.
//
//    Name: MYY10251_IA                      Date: 2024/01/09
//    User: AliAl                            Time: 13:42:05
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
  /// Internal data view storage for: MYY10251_IA
  /// </summary>
  [Serializable]
  public class MYY10251_IA : ViewBase, IImportView
  {
    private static MYY10251_IA[] freeArray = new MYY10251_IA[30];
    private static int countFree = 0;
    
    // Entity View: IMP_FILTER
    //        Type: IYY1_LIST
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    private char _ImpFilterIyy1ListSortOption_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListSortOption
    /// </summary>
    public char ImpFilterIyy1ListSortOption_AS {
      get {
        return(_ImpFilterIyy1ListSortOption_AS);
      }
      set {
        _ImpFilterIyy1ListSortOption_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListSortOption
    /// Domain: Text
    /// Length: 3
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListSortOption;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListSortOption
    /// </summary>
    public string ImpFilterIyy1ListSortOption {
      get {
        return(_ImpFilterIyy1ListSortOption);
      }
      set {
        _ImpFilterIyy1ListSortOption = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    private char _ImpFilterIyy1ListScrollType_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollType
    /// </summary>
    public char ImpFilterIyy1ListScrollType_AS {
      get {
        return(_ImpFilterIyy1ListScrollType_AS);
      }
      set {
        _ImpFilterIyy1ListScrollType_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollType
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListScrollType;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollType
    /// </summary>
    public string ImpFilterIyy1ListScrollType {
      get {
        return(_ImpFilterIyy1ListScrollType);
      }
      set {
        _ImpFilterIyy1ListScrollType = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    private char _ImpFilterIyy1ListListDirection_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListListDirection
    /// </summary>
    public char ImpFilterIyy1ListListDirection_AS {
      get {
        return(_ImpFilterIyy1ListListDirection_AS);
      }
      set {
        _ImpFilterIyy1ListListDirection_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListListDirection
    /// Domain: Text
    /// Length: 1
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ListListDirection;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListListDirection
    /// </summary>
    public string ImpFilterIyy1ListListDirection {
      get {
        return(_ImpFilterIyy1ListListDirection);
      }
      set {
        _ImpFilterIyy1ListListDirection = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    private char _ImpFilterIyy1ListScrollAmount_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public char ImpFilterIyy1ListScrollAmount_AS {
      get {
        return(_ImpFilterIyy1ListScrollAmount_AS);
      }
      set {
        _ImpFilterIyy1ListScrollAmount_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListScrollAmount
    /// Domain: Number
    /// Length: 5
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterIyy1ListScrollAmount;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListScrollAmount
    /// </summary>
    public int ImpFilterIyy1ListScrollAmount {
      get {
        return(_ImpFilterIyy1ListScrollAmount);
      }
      set {
        _ImpFilterIyy1ListScrollAmount = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    private char _ImpFilterIyy1ListOrderByFieldNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public char ImpFilterIyy1ListOrderByFieldNum_AS {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum_AS);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// Domain: Number
    /// Length: 1
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private short _ImpFilterIyy1ListOrderByFieldNum;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ListOrderByFieldNum
    /// </summary>
    public short ImpFilterIyy1ListOrderByFieldNum {
      get {
        return(_ImpFilterIyy1ListOrderByFieldNum);
      }
      set {
        _ImpFilterIyy1ListOrderByFieldNum = value;
      }
    }
    // Entity View: IMP_FROM
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1ChildCinstanceId
    /// </summary>
    private char _ImpFromIyy1ChildCinstanceId_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1ChildCinstanceId
    /// </summary>
    public char ImpFromIyy1ChildCinstanceId_AS {
      get {
        return(_ImpFromIyy1ChildCinstanceId_AS);
      }
      set {
        _ImpFromIyy1ChildCinstanceId_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1ChildCinstanceId
    /// Domain: Timestamp
    /// Length: 20
    /// </summary>
    private string _ImpFromIyy1ChildCinstanceId;
    /// <summary>
    /// Attribute for: ImpFromIyy1ChildCinstanceId
    /// </summary>
    public string ImpFromIyy1ChildCinstanceId {
      get {
        return(_ImpFromIyy1ChildCinstanceId);
      }
      set {
        _ImpFromIyy1ChildCinstanceId = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1ChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFromIyy1ChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1ChildCparentPkeyAttrText
    /// </summary>
    public char ImpFromIyy1ChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFromIyy1ChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFromIyy1ChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1ChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFromIyy1ChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFromIyy1ChildCparentPkeyAttrText
    /// </summary>
    public string ImpFromIyy1ChildCparentPkeyAttrText {
      get {
        return(_ImpFromIyy1ChildCparentPkeyAttrText);
      }
      set {
        _ImpFromIyy1ChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFromIyy1ChildCkeyAttrNum
    /// </summary>
    private char _ImpFromIyy1ChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFromIyy1ChildCkeyAttrNum
    /// </summary>
    public char ImpFromIyy1ChildCkeyAttrNum_AS {
      get {
        return(_ImpFromIyy1ChildCkeyAttrNum_AS);
      }
      set {
        _ImpFromIyy1ChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFromIyy1ChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFromIyy1ChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFromIyy1ChildCkeyAttrNum
    /// </summary>
    public int ImpFromIyy1ChildCkeyAttrNum {
      get {
        return(_ImpFromIyy1ChildCkeyAttrNum);
      }
      set {
        _ImpFromIyy1ChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER_START
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStartIyy1ChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFilterStartIyy1ChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStartIyy1ChildCparentPkeyAttrText
    /// </summary>
    public char ImpFilterStartIyy1ChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFilterStartIyy1ChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFilterStartIyy1ChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStartIyy1ChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStartIyy1ChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStartIyy1ChildCparentPkeyAttrText
    /// </summary>
    public string ImpFilterStartIyy1ChildCparentPkeyAttrText {
      get {
        return(_ImpFilterStartIyy1ChildCparentPkeyAttrText);
      }
      set {
        _ImpFilterStartIyy1ChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStartIyy1ChildCkeyAttrNum
    /// </summary>
    private char _ImpFilterStartIyy1ChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStartIyy1ChildCkeyAttrNum
    /// </summary>
    public char ImpFilterStartIyy1ChildCkeyAttrNum_AS {
      get {
        return(_ImpFilterStartIyy1ChildCkeyAttrNum_AS);
      }
      set {
        _ImpFilterStartIyy1ChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStartIyy1ChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterStartIyy1ChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFilterStartIyy1ChildCkeyAttrNum
    /// </summary>
    public int ImpFilterStartIyy1ChildCkeyAttrNum {
      get {
        return(_ImpFilterStartIyy1ChildCkeyAttrNum);
      }
      set {
        _ImpFilterStartIyy1ChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER_STOP
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStopIyy1ChildCparentPkeyAttrText
    /// </summary>
    private char _ImpFilterStopIyy1ChildCparentPkeyAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStopIyy1ChildCparentPkeyAttrText
    /// </summary>
    public char ImpFilterStopIyy1ChildCparentPkeyAttrText_AS {
      get {
        return(_ImpFilterStopIyy1ChildCparentPkeyAttrText_AS);
      }
      set {
        _ImpFilterStopIyy1ChildCparentPkeyAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStopIyy1ChildCparentPkeyAttrText
    /// Domain: Text
    /// Length: 5
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterStopIyy1ChildCparentPkeyAttrText;
    /// <summary>
    /// Attribute for: ImpFilterStopIyy1ChildCparentPkeyAttrText
    /// </summary>
    public string ImpFilterStopIyy1ChildCparentPkeyAttrText {
      get {
        return(_ImpFilterStopIyy1ChildCparentPkeyAttrText);
      }
      set {
        _ImpFilterStopIyy1ChildCparentPkeyAttrText = value;
      }
    }
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterStopIyy1ChildCkeyAttrNum
    /// </summary>
    private char _ImpFilterStopIyy1ChildCkeyAttrNum_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterStopIyy1ChildCkeyAttrNum
    /// </summary>
    public char ImpFilterStopIyy1ChildCkeyAttrNum_AS {
      get {
        return(_ImpFilterStopIyy1ChildCkeyAttrNum_AS);
      }
      set {
        _ImpFilterStopIyy1ChildCkeyAttrNum_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterStopIyy1ChildCkeyAttrNum
    /// Domain: Number
    /// Length: 6
    /// Decimal Places: 0
    /// Decimal Precision: N
    /// </summary>
    private int _ImpFilterStopIyy1ChildCkeyAttrNum;
    /// <summary>
    /// Attribute for: ImpFilterStopIyy1ChildCkeyAttrNum
    /// </summary>
    public int ImpFilterStopIyy1ChildCkeyAttrNum {
      get {
        return(_ImpFilterStopIyy1ChildCkeyAttrNum);
      }
      set {
        _ImpFilterStopIyy1ChildCkeyAttrNum = value;
      }
    }
    // Entity View: IMP_FILTER
    //        Type: IYY1_CHILD
    /// <summary>
    /// Internal storage for attribute missing flag for: ImpFilterIyy1ChildCsearchAttrText
    /// </summary>
    private char _ImpFilterIyy1ChildCsearchAttrText_AS;
    /// <summary>
    /// Attribute missing flag for: ImpFilterIyy1ChildCsearchAttrText
    /// </summary>
    public char ImpFilterIyy1ChildCsearchAttrText_AS {
      get {
        return(_ImpFilterIyy1ChildCsearchAttrText_AS);
      }
      set {
        _ImpFilterIyy1ChildCsearchAttrText_AS = value;
      }
    }
    /// <summary>
    /// Internal storage, attribute for: ImpFilterIyy1ChildCsearchAttrText
    /// Domain: Text
    /// Length: 25
    /// Varying Length: N
    /// </summary>
    private string _ImpFilterIyy1ChildCsearchAttrText;
    /// <summary>
    /// Attribute for: ImpFilterIyy1ChildCsearchAttrText
    /// </summary>
    public string ImpFilterIyy1ChildCsearchAttrText {
      get {
        return(_ImpFilterIyy1ChildCsearchAttrText);
      }
      set {
        _ImpFilterIyy1ChildCsearchAttrText = value;
      }
    }
    /// <summary>
    /// Default Constructor
    /// </summary>
    
    public MYY10251_IA(  )
    {
      Reset(  );
    }
    /// <summary>
    /// Copy Constructor
    /// </summary>
    
    public MYY10251_IA( MYY10251_IA orig )
    {
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromIyy1ChildCinstanceId_AS = orig.ImpFromIyy1ChildCinstanceId_AS;
      ImpFromIyy1ChildCinstanceId = orig.ImpFromIyy1ChildCinstanceId;
      ImpFromIyy1ChildCparentPkeyAttrText_AS = orig.ImpFromIyy1ChildCparentPkeyAttrText_AS;
      ImpFromIyy1ChildCparentPkeyAttrText = orig.ImpFromIyy1ChildCparentPkeyAttrText;
      ImpFromIyy1ChildCkeyAttrNum_AS = orig.ImpFromIyy1ChildCkeyAttrNum_AS;
      ImpFromIyy1ChildCkeyAttrNum = orig.ImpFromIyy1ChildCkeyAttrNum;
      ImpFilterStartIyy1ChildCparentPkeyAttrText_AS = orig.ImpFilterStartIyy1ChildCparentPkeyAttrText_AS;
      ImpFilterStartIyy1ChildCparentPkeyAttrText = orig.ImpFilterStartIyy1ChildCparentPkeyAttrText;
      ImpFilterStartIyy1ChildCkeyAttrNum_AS = orig.ImpFilterStartIyy1ChildCkeyAttrNum_AS;
      ImpFilterStartIyy1ChildCkeyAttrNum = orig.ImpFilterStartIyy1ChildCkeyAttrNum;
      ImpFilterStopIyy1ChildCparentPkeyAttrText_AS = orig.ImpFilterStopIyy1ChildCparentPkeyAttrText_AS;
      ImpFilterStopIyy1ChildCparentPkeyAttrText = orig.ImpFilterStopIyy1ChildCparentPkeyAttrText;
      ImpFilterStopIyy1ChildCkeyAttrNum_AS = orig.ImpFilterStopIyy1ChildCkeyAttrNum_AS;
      ImpFilterStopIyy1ChildCkeyAttrNum = orig.ImpFilterStopIyy1ChildCkeyAttrNum;
      ImpFilterIyy1ChildCsearchAttrText_AS = orig.ImpFilterIyy1ChildCsearchAttrText_AS;
      ImpFilterIyy1ChildCsearchAttrText = orig.ImpFilterIyy1ChildCsearchAttrText;
    }
    /// <summary>
    /// Static instance creator function
    /// </summary>
    
    public static MYY10251_IA GetInstance(  )
    {
      if ( countFree == 0 )
      {
        return(new MYY10251_IA());
      }
      else 
      {
        lock (freeArray)
        {
          if ( countFree == 0 )
          {
            return(new MYY10251_IA());
          }
          else 
          {
            MYY10251_IA result = freeArray[--countFree];
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
      return(new MYY10251_IA(this));
    }
    /// <summary>
    /// Resets all properties to the defaults.
    /// </summary>
    
    public void Reset(  )
    {
      ImpFilterIyy1ListSortOption_AS = ' ';
      ImpFilterIyy1ListSortOption = "   ";
      ImpFilterIyy1ListScrollType_AS = ' ';
      ImpFilterIyy1ListScrollType = " ";
      ImpFilterIyy1ListListDirection_AS = ' ';
      ImpFilterIyy1ListListDirection = " ";
      ImpFilterIyy1ListScrollAmount_AS = ' ';
      ImpFilterIyy1ListScrollAmount = 0;
      ImpFilterIyy1ListOrderByFieldNum_AS = ' ';
      ImpFilterIyy1ListOrderByFieldNum = 0;
      ImpFromIyy1ChildCinstanceId_AS = ' ';
      ImpFromIyy1ChildCinstanceId = "00000000000000000000";
      ImpFromIyy1ChildCparentPkeyAttrText_AS = ' ';
      ImpFromIyy1ChildCparentPkeyAttrText = "     ";
      ImpFromIyy1ChildCkeyAttrNum_AS = ' ';
      ImpFromIyy1ChildCkeyAttrNum = 0;
      ImpFilterStartIyy1ChildCparentPkeyAttrText_AS = ' ';
      ImpFilterStartIyy1ChildCparentPkeyAttrText = "     ";
      ImpFilterStartIyy1ChildCkeyAttrNum_AS = ' ';
      ImpFilterStartIyy1ChildCkeyAttrNum = 0;
      ImpFilterStopIyy1ChildCparentPkeyAttrText_AS = ' ';
      ImpFilterStopIyy1ChildCparentPkeyAttrText = "     ";
      ImpFilterStopIyy1ChildCkeyAttrNum_AS = ' ';
      ImpFilterStopIyy1ChildCkeyAttrNum = 0;
      ImpFilterIyy1ChildCsearchAttrText_AS = ' ';
      ImpFilterIyy1ChildCsearchAttrText = "                         ";
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
      this.CopyFrom((MYY10251_IA) orig);
    }
    
    /// <summary>
    /// Sets the current instance based on the passed view.
    /// </summary>
    public void CopyFrom( MYY10251_IA orig )
    {
      ImpFilterIyy1ListSortOption_AS = orig.ImpFilterIyy1ListSortOption_AS;
      ImpFilterIyy1ListSortOption = orig.ImpFilterIyy1ListSortOption;
      ImpFilterIyy1ListScrollType_AS = orig.ImpFilterIyy1ListScrollType_AS;
      ImpFilterIyy1ListScrollType = orig.ImpFilterIyy1ListScrollType;
      ImpFilterIyy1ListListDirection_AS = orig.ImpFilterIyy1ListListDirection_AS;
      ImpFilterIyy1ListListDirection = orig.ImpFilterIyy1ListListDirection;
      ImpFilterIyy1ListScrollAmount_AS = orig.ImpFilterIyy1ListScrollAmount_AS;
      ImpFilterIyy1ListScrollAmount = orig.ImpFilterIyy1ListScrollAmount;
      ImpFilterIyy1ListOrderByFieldNum_AS = orig.ImpFilterIyy1ListOrderByFieldNum_AS;
      ImpFilterIyy1ListOrderByFieldNum = orig.ImpFilterIyy1ListOrderByFieldNum;
      ImpFromIyy1ChildCinstanceId_AS = orig.ImpFromIyy1ChildCinstanceId_AS;
      ImpFromIyy1ChildCinstanceId = orig.ImpFromIyy1ChildCinstanceId;
      ImpFromIyy1ChildCparentPkeyAttrText_AS = orig.ImpFromIyy1ChildCparentPkeyAttrText_AS;
      ImpFromIyy1ChildCparentPkeyAttrText = orig.ImpFromIyy1ChildCparentPkeyAttrText;
      ImpFromIyy1ChildCkeyAttrNum_AS = orig.ImpFromIyy1ChildCkeyAttrNum_AS;
      ImpFromIyy1ChildCkeyAttrNum = orig.ImpFromIyy1ChildCkeyAttrNum;
      ImpFilterStartIyy1ChildCparentPkeyAttrText_AS = orig.ImpFilterStartIyy1ChildCparentPkeyAttrText_AS;
      ImpFilterStartIyy1ChildCparentPkeyAttrText = orig.ImpFilterStartIyy1ChildCparentPkeyAttrText;
      ImpFilterStartIyy1ChildCkeyAttrNum_AS = orig.ImpFilterStartIyy1ChildCkeyAttrNum_AS;
      ImpFilterStartIyy1ChildCkeyAttrNum = orig.ImpFilterStartIyy1ChildCkeyAttrNum;
      ImpFilterStopIyy1ChildCparentPkeyAttrText_AS = orig.ImpFilterStopIyy1ChildCparentPkeyAttrText_AS;
      ImpFilterStopIyy1ChildCparentPkeyAttrText = orig.ImpFilterStopIyy1ChildCparentPkeyAttrText;
      ImpFilterStopIyy1ChildCkeyAttrNum_AS = orig.ImpFilterStopIyy1ChildCkeyAttrNum_AS;
      ImpFilterStopIyy1ChildCkeyAttrNum = orig.ImpFilterStopIyy1ChildCkeyAttrNum;
      ImpFilterIyy1ChildCsearchAttrText_AS = orig.ImpFilterIyy1ChildCsearchAttrText_AS;
      ImpFilterIyy1ChildCsearchAttrText = orig.ImpFilterIyy1ChildCsearchAttrText;
    }
  }
}
